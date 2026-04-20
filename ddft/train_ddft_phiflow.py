import pde
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import glob
import os
import argparse
from tqdm import tqdm

from neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from phi.torch.flow import *

#################
##### Setup #####
#################

parser = argparse.ArgumentParser(description='End-to-End Differentiable Training for Underdamped DDFT')
parser.add_argument('--run_id', type=str, default='tdamp10_dz32')
parser.add_argument('--md_path', type=str, required=True, help='Path to MD .h5 trajectory files')
parser.add_argument('--type', type=str, default='c2', help='Base model type (c1, c2)')
parser.add_argument('--dr', type=float, default=1/32, help='Grid spacing of the MD simulation')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')

# Physics Params (Must match the MD simulation!)
parser.add_argument('--k', type=float, default=1)
parser.add_argument('--T', type=float, default=2)
parser.add_argument('--m', type=float, default=1)
parser.add_argument('--gamma', type=float, default=2)
parser.add_argument('--eta', type=float, default=0.01)
parser.add_argument('--dt', type=float, default=1e-4, help='Internal PDE time-step')

args = parser.parse_args()

os.makedirs("models/finetuned", exist_ok=True)

# ==========================================
# 1. Load MD Trajectory Data
# ==========================================
print(f"Loading MD data from {args.md_path}...")
file_pattern = f"{args.md_path}/collection.*.h5"
md_file_list = sorted(glob.glob(file_pattern))

if len(md_file_list) < 2:
    raise FileNotFoundError("Need at least 2 MD files to compute a time step!")

md_data = []
for file_path in md_file_list:
    with h5py.File(file_path, 'r') as f:
        rho_md = np.array(f['n']).flatten()
        v_md = np.array(f['vz']).flatten()
        if len(md_data) == 0:
            V_ext_np = np.array(f['V']).flatten()
            r_np = np.array(f['r']).flatten()
        md_data.append({'rho': rho_md, 'v': v_md})

dz = r_np[1] - r_np[0]
box_length = r_np[-1] - r_np[0]
domain_resolution = r_np.shape[0]
dx = float(box_length / domain_resolution)

# Figure out how many PDE steps make up one MD save interval
# Assuming your MD files are saved e.g. every 0.01 time units
time_between_md_saves = 0.01 
pde_steps_per_md_save = int(time_between_md_saves / args.dt)

# ==========================================
# 2. Load Pre-trained Model & Setup Optimizer
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

print("Loading base model...")
if int(1/args.dr) == 100:
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=f"models/{args.type}_dz100.ckpt", map_location=device)
else:
    model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=f"models/{args.type}_dz{int(1/args.dr)}.ckpt", map_location=device)

# Set to train mode (enables gradients on weights)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Define External Potential on Grid
V_tensor_pt = torch.tensor(V_ext_np, dtype=torch.float32, device=device)
V_ext = CenteredGrid(math.tensor(V_tensor_pt, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))

# ==========================================
# 3. Differentiable Physics Functions
# ==========================================

def dft_U_phiflow_train(rho_grid):
    """
    Computes δF_exc/δρ while keeping the computational graph ALIVE 
    so we can backpropagate the PDE error into the NN weights.
    """
    rho_pt = rho_grid.values.native()
    rho_tensor = rho_pt.view(1, 1, -1).float().requires_grad_(True)
    
    F_pred = model(rho_tensor)
    
    # CRITICAL CHANGE: create_graph=True
    dFdrho = torch.autograd.grad(F_pred, rho_tensor, create_graph=True)[0] / dz
    
    dFdrho_1d = dFdrho.squeeze()
    dFdrho_phi = math.tensor(dFdrho_1d, spatial('x'))
    return CenteredGrid(dFdrho_phi, extrapolation.PERIODIC, bounds=rho_grid.bounds)


def step_diff(rho, v, dt, gamma, eta, m, k, T, V_ext):
    """ Differentiable step function (No kill-switches or detached numpy arrays inside!) """
    rho_faces = rho.at(v)
    flux_centered = rho_faces * v

    grad_rho = field.spatial_gradient(rho)      
    D_art = 0.5 * dx * math.abs(v)              
    flux_diffusive = - D_art * grad_rho          
    total_flux = flux_centered + flux_diffusive
    
    rho_next = rho - dt * field.divergence(total_flux)
    
    # Differentiable clamp to prevent log(0) issues
    safe_rho_vals = math.maximum(rho_next.values, 1e-6)
    rho_next = rho_next.with_values(safe_rho_vals)
    
    v_advected = advect.semi_lagrangian(v, v, dt)
    rho_faces_next = rho_next.at(v)
    
    fric_term = -gamma * v
    visc_term = (eta / (m * rho_faces_next)) * field.laplace(v)
    pressure_term = - (k * T / (m * rho_faces_next)) * field.spatial_gradient(rho_next)
    vext_term = - (1.0 / m) * field.spatial_gradient(V_ext)
    
    # Predict Superadiabatic-corrected free energy gradient
    dFdrho_grid = dft_U_phiflow_train(rho_next)
    exc_term = - (1.0 / m) * field.spatial_gradient(dFdrho_grid)
    
    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + exc_term)
    
    return rho_next, v_next

# ==========================================
# 4. Training Loop
# ==========================================

print("Starting End-to-End Fine-Tuning...")

for epoch in range(args.epochs):
    epoch_loss = 0.0
    
    # Iterate through the MD trajectory in pairs: (t) -> target(t + save_interval)
    # We use tqdm on the inner loop to show progress through the trajectory
    pbar = tqdm(range(len(md_data) - 1), desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for i in pbar:
        # Get start state from MD
        current_md = md_data[i]
        target_md = md_data[i+1]
        
        # Initialize PhiFlow grids with PyTorch tensors requiring grad
        rho_tensor = torch.tensor(current_md['rho'], dtype=torch.float32, device=device, requires_grad=True)
        v_tensor = torch.tensor(current_md['v'], dtype=torch.float32, device=device, requires_grad=True)
        
        rho_phi = CenteredGrid(math.tensor(rho_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
        v_phi = StaggeredGrid(math.tensor(v_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
        
        # Unroll the PDE for 'N' steps to reach the next MD save point
        for step_pde in range(pde_steps_per_md_save):
            rho_phi, v_phi = step_diff(rho_phi, v_phi, args.dt, args.gamma, args.eta, args.m, args.k, args.T, V_ext)
            
        # Extract the final predicted PyTorch tensors
        rho_pred = rho_phi.values.native('x')
        v_pred = v_phi.values.vector['x'].native('x')
        
        # Target tensors
        rho_target = torch.tensor(target_md['rho'], dtype=torch.float32, device=device)
        v_target = torch.tensor(target_md['v'], dtype=torch.float32, device=device)
        
        # Compute Loss (Mean Squared Error)
        # We scale velocity loss to put it roughly on the same magnitude as density
        loss_rho = F.mse_loss(rho_pred, rho_target)
        loss_v = F.mse_loss(v_pred, v_target) 
        
        loss_scale = 1000.0 # scale the loss up
        loss = (loss_rho + (loss_v * 0.1)) * loss_scale
        
        # Backpropagate through time
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients during PDE backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
    avg_loss = epoch_loss / (len(md_data) - 1)
    print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.6f}")

# ==========================================
# 5. Save the Fine-Tuned Copy
# ==========================================
save_path = f"models/finetuned/{args.type}_finetuned_{args.run_id}.ckpt"

# Because the model was loaded via PyTorch Lightning, we extract its state_dict 
# and save it natively so it can be loaded later.
model.cpu()
torch.save({'state_dict': model.state_dict()}, save_path)
print(f"Training Complete! New model successfully saved to: {save_path}")
