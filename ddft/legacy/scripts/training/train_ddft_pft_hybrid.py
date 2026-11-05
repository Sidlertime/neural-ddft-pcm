import os
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from phi.torch.flow import *

# Import our custom modules
from neural_ddft.data.dataset import MDWindowDataset
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.models.neural_free_energy import NeuralFreeEnergy

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, required=True)
parser.add_argument('--md_path', type=str, required=True)
parser.add_argument('--base_ckpt', type=str, required=True, help="Path to frozen c2 model")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 1. Physics Setup
# ==========================================
k, T, m = 1.0, 2.0, 1.0
gamma = 0.1
eta =  0.78
dt = 1e-4
time_between_md_saves = 0.01 
pde_steps = int(time_between_md_saves / dt)

# ==========================================
# 2. Load Models
# ==========================================
# A. The Frozen Equilibrium Model (cDFT)
base_model = NeuralFreeEnergy.load_from_checkpoint(args.base_ckpt, map_location=device)
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False # FREEZE IT!

# B. The Trainable Superadiabatic Model (PFT)
sup_model = SuperadiabaticNet().to(device)
optimizer = torch.optim.Adam(sup_model.parameters(), lr=args.lr)

# ==========================================
# 3. The Differentiable PDE Step
# ==========================================
def step_diff_batch(rho, v, dx, V_ext):
    """ Differentiable PhiFlow step with BATCH support """
    rho_faces = rho.at(v)
    
    # 1. Continuity Eq
    flux_centered = rho_faces * v
    grad_rho = field.spatial_gradient(rho)      
    D_art = 0.5 * dx * math.abs(v)              
    flux_diffusive = - D_art * grad_rho          
    rho_next = rho - dt * field.divergence(flux_centered + flux_diffusive)
    rho_next = rho_next.with_values(math.maximum(rho_next.values, 1e-6))
    
    # 2. Velocity Eq
    v_advected = advect.semi_lagrangian(v, v, dt)
    rho_faces_next = rho_next.at(v)
    
    fric_term = -gamma * v
    visc_term = (eta / (m * rho_faces_next)) * field.laplace(v)
    pressure_term = - (k * T / (m * rho_faces_next)) * field.spatial_gradient(rho_next)
    vext_term = - (1.0 / m) * field.spatial_gradient(V_ext)

    if math.is_finite(rho_next.values).any == False:
        print("CRITICAL PDE FAILURE: Density became NaN or Inf!")
        print(f"Max Velocity in previous step: {math.max(math.abs(v.values))}")
        raise ValueError("PDE CFL Explosion.")
    
    # --- The Hybrid Force ---
    # A. Frozen Equilibrium Force
    rho_tensor = rho_next.values.native('batch,x').unsqueeze(1).detach().requires_grad_(True)
    with torch.enable_grad():
        F_pred = base_model(rho_tensor)
        # create_graph=True ensures the PDE solver can backpropagate through this derivative later!
        dFdrho_tensor = torch.autograd.grad(F_pred.sum(), rho_tensor, create_graph=False)[0] / dx
        # --- NEW: THERMODYNAMIC FORCE CLAMP ---
        dFdrho_tensor = torch.clamp(dFdrho_tensor, min=-100.0, max=100.0)
    
    exc_term = - (1.0 / m) * field.spatial_gradient(CenteredGrid(math.tensor(dFdrho_tensor.squeeze(), batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho.bounds))
    
    # B. Trainable Superadiabatic Force
    # Pass instantaneous batch tensors to the new CNN
    f_sup_tensor = sup_model(rho_next.values.native('batch,x'), v.values.vector['x'].native('batch,x'))
    f_sup_grid = StaggeredGrid(math.tensor(f_sup_tensor, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho.bounds)
    
    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + exc_term + f_sup_grid)
    # --- NEW: PDE Safety Valve ---
    v_clamped = math.clip(v_next.values, -100.0, 100.0) 
    v_next = v_next.with_values(v_clamped)
    
    return rho_next, v_next

# ==========================================
# 4. Curriculum Training Loop
# ==========================================
sigma_smooth = 4.0
save_every = 5
for epoch in range(args.epochs):
    # --- Curriculum Scheduler ---
    if epoch < 10:
        window_size, batch_size = 5, 32   # Heavy Teacher Forcing
        stride, loss_v_scale = 2, 0.05
    elif epoch < 30:
        window_size, batch_size = 25, 8   # Autoregressive Transition
        stride, loss_v_scale = 10, 0.25
    else:
        window_size, batch_size = 50, 2   # Full Rollout
        stride, loss_v_scale = 25, 1.0
        
    dataset = MDWindowDataset(args.md_path, window_size, stride=stride, sigma_smooth=sigma_smooth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Velocity Loss Scale
    #progress = epoch / max(1, args.epochs - 1)
    #loss_v_scale = 0.1 + (max(0.0, min(1.0, (progress - 0.1) / 0.8)) * 0.9)

    # Accumulators for RAW unscaled physical metrics
    epoch_raw_rho_loss = 0.0
    epoch_raw_v_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} (Win: {window_size})")
    
    for batch_data in pbar:
        # batch_data['rho'] shape: (Batch, Window+1, X)
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_raw_rho = 0.0
        batch_raw_v = 0.0

        # setup V_ext
        current_batch_size = batch_data['rho'].shape[0]
        V_base = dataset.V_ext.to(device)
        V_batched = V_base.unsqueeze(0).expand(current_batch_size, -1)
        V_ext_grid = CenteredGrid(
            math.tensor(V_batched, batch('batch'), spatial('x')), 
            extrapolation.PERIODIC, 
            bounds=Box(x=dataset.box_length)
        )
        
        # Initialize physical state at t=0 of the window
        # We explicitly define the 'batch' dimension for PhiFlow
        rho_init = batch_data['rho'][:, 0, :].to(device).requires_grad_(True)
        v_init = batch_data['v'][:, 0, :].to(device).requires_grad_(True)
        
        rho_phi = CenteredGrid(math.tensor(rho_init, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=Box(x=dataset.box_length))
        v_phi = StaggeredGrid(math.tensor(v_init, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=Box(x=dataset.box_length))
        
        # Unroll the window
        for w in range(1, window_size + 1):
            # Step the PDE
            for _ in range(pde_steps):
                rho_phi, v_phi = step_diff_batch(rho_phi, v_phi, dataset.dr, V_ext_grid)
                
            # Calculate step loss against the exact MD target at this window index
            rho_pred = rho_phi.values.native('batch,x')
            v_pred = v_phi.values.vector['x'].native('batch,x')
            
            rho_target = batch_data['rho'][:, w, :].to(device)
            v_target = batch_data['v'][:, w, :].to(device)

            step_rho_loss = F.mse_loss(rho_pred, rho_target)
            step_v_loss = F.mse_loss(v_pred, v_target)

            batch_raw_rho += step_rho_loss.item()
            batch_raw_v += step_v_loss.item()
            
            step_loss_total = step_rho_loss + (loss_v_scale * step_v_loss)
            batch_loss = batch_loss + step_loss_total
            
        # Backpropagate the whole window
        scaled_loss = batch_loss * 10000.0
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(sup_model.parameters(), 1.0)
        optimizer.step()
        
        epoch_raw_rho_loss += (batch_raw_rho / window_size)
        epoch_raw_v_loss += (batch_raw_v / window_size)
        
        pbar.set_postfix({
            'L_rho': f"{epoch_raw_rho_loss / (pbar.n + 1):.6f}", 
            'L_v': f"{epoch_raw_v_loss / (pbar.n + 1):.6f}",
            'scale': f"{loss_v_scale:.2f}"
        })

    # epoch checkpointing
    if (epoch + 1) % save_every == 0 or epoch == args.epochs - 1:
        sup_model.cpu() # Move to CPU for safe saving
        ckpt_name = f"models/checkpoints/{args.run_id}_epoch_{epoch+1}.ckpt"
        torch.save(sup_model.state_dict(), ckpt_name)
        sup_model.to(device) # Put it back on the GPU for the next epoch!
        print(f"\n[Checkpoint Saved]: {ckpt_name}")

# Save the new PFT Force Model
sup_model.cpu()
torch.save(sup_model.state_dict(), f"models/{args.run_id}.ckpt")