import pde
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import glob
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from phi.torch.flow import *
from scipy.ndimage import gaussian_filter1d

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

sigma_smooth = 4.0
md_data = []
md_data_raw = []
for file_path in md_file_list:
    with h5py.File(file_path, 'r') as f:
        rho_md = np.array(f['n']).flatten()
        v_md = np.array(f['vz']).flatten()

        # Apply Periodic Gaussian Smoothing
        rho_smooth = gaussian_filter1d(rho_md, sigma=sigma_smooth, mode='wrap')
        v_smooth = gaussian_filter1d(v_md, sigma=sigma_smooth, mode='wrap')

        if len(md_data) == 0:
            V_ext_np = np.array(f['V']).flatten()
            r_np = np.array(f['r']).flatten()

        md_data_raw.append({'rho': rho_md, 'v': v_md})
        md_data.append({'rho': rho_smooth, 'v': v_smooth})

'''
print("Generating MD trajectory animation...")

# Setup the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MD Trajectory (Smoothed Target Data)')

# Initialize the lines
line_rho, = ax1.plot([], [], 'b-', lw=2)
line_v, = ax2.plot([], [], 'b-', lw=2)
line_rho_raw, = ax1.plot([], [], 'r--', lw=2)
line_v_raw, = ax2.plot([], [], 'r--', lw=2)

# Set axis limits
ax1.set_xlim(r_np[0], r_np[-1])
ax1.set_ylim(0, np.max([np.max(d['rho']) for d in [*md_data, *md_data_raw]]) * 1.1)
ax1.set_title('Density (rho)')
ax1.set_xlabel('z')

ax2.set_xlim(r_np[0], r_np[-1])
v_max = np.max([np.max(np.abs(d['v'])) for d in [*md_data, *md_data_raw]])
ax2.set_ylim(-v_max * 1.1, v_max * 1.1)
ax2.set_title('Velocity (v)')
ax2.set_xlabel('z')

# Add a time/frame counter text
time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, va='top')

# Initialization function
def init():
    line_rho.set_data([], [])
    line_v.set_data([], [])
    line_rho_raw.set_data([], [])
    line_v_raw.set_data([], [])
    time_text.set_text('')
    return line_rho, line_v, line_rho_raw, line_v_raw, time_text

# Animation update function
def animate(i):
    line_rho.set_data(r_np, md_data[i]['rho'])
    line_v.set_data(r_np, md_data[i]['v'])
    line_rho_raw.set_data(r_np, md_data_raw[i]['rho'])
    line_v_raw.set_data(r_np, md_data_raw[i]['v'])
    
    # Assuming each frame is separated by 'time_between_md_saves' (e.g., 0.01)
    current_time = i * 0.01 
    time_text.set_text(f'Time: {current_time:.2f}')
    return line_rho, line_v, line_rho_raw, line_v_raw, time_text

# Create the animation
anim = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(md_data), interval=50, blit=True
)

# Save the animation (requires ffmpeg installed)
save_path = f"md_smoothed{sigma_smooth}_trajectory.mp4"
anim.save(save_path, writer='ffmpeg', fps=30)
print(f"Animation saved to {save_path}")

plt.close(fig) # Prevent the plot from hanging the terminal
'''

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

window_size = 50

for epoch in range(args.epochs):
    epoch_loss = 0.0
    num_windows = 0

    #loss_v_scale = 0.1
    progress = epoch / max(1, args.epochs - 1)  # progression of epochs (0.0 to 1.0)
    ramp_progress = (progress - 0.1) / 0.8
    ramp_progress = max(0.0, min(1.0, ramp_progress))   # start rising at 0.1 and maximizes at 0.9
    loss_v_scale = 0.1 + (ramp_progress * 0.9)  # 0.1 for the first 10% progress, ramps to 1.0 at 90% and stays there to focus on learning velocity towards the end
    
    # Iterate through the MD trajectory in pairs: (t) -> target(t + save_interval)
    # We use tqdm on the inner loop to show progress through the trajectory
    pbar = tqdm(range(0, len(md_data) - 1, window_size), desc=f"Epoch {epoch+1}/{args.epochs}")

    # continous run (like production)
    start_md = md_data[0]
    rho_tensor = torch.tensor(start_md['rho'], dtype=torch.float32, device=device, requires_grad=True)
    v_tensor = torch.tensor(start_md['v'], dtype=torch.float32, device=device, requires_grad=True)
    
    rho_phi = CenteredGrid(math.tensor(rho_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
    v_phi = StaggeredGrid(math.tensor(v_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
    
    for start_idx in pbar:
        # Determine the end of the current window (handle the leftover tail of the dataset)
        end_idx = min(start_idx + window_size, len(md_data) - 1)

        optimizer.zero_grad()
        window_loss = 0.0

        # INNER LOOP: Unroll physics inside the window
        for i in range(start_idx, end_idx):

            # Get start state from MD
            #current_md = md_data[i]
            target_md = md_data[i+1]
            
            # Initialize PhiFlow grids with PyTorch tensors requiring grad
            #rho_tensor = torch.tensor(current_md['rho'], dtype=torch.float32, device=device, requires_grad=True)
            #v_tensor = torch.tensor(current_md['v'], dtype=torch.float32, device=device, requires_grad=True)
            
            #rho_phi = CenteredGrid(math.tensor(rho_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
            #v_phi = StaggeredGrid(math.tensor(v_tensor, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
            
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
            window_loss = window_loss + (loss_rho + (loss_v * loss_v_scale)) * loss_scale
        
        # Backpropagate through time
        window_loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients during PDE backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += window_loss.item() / (end_idx - start_idx) # Average loss per step
        num_windows += 1
        pbar.set_postfix({'loss': f"{epoch_loss/num_windows:.6f}"})

        # continous run
        rho_detached = rho_pred.detach().clone().requires_grad_(True)
        v_detached = v_pred.detach().clone().requires_grad_(True)
        
        # Repackage them into clean PhiFlow grids for the next iteration
        rho_phi = CenteredGrid(math.tensor(rho_detached, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
        v_phi = StaggeredGrid(math.tensor(v_detached, spatial('x')), extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
        
    print(f"Epoch {epoch+1} Completed | Average Step Loss: {epoch_loss / num_windows:.6f}")

# ==========================================
# 5. Save the Fine-Tuned Copy
# ==========================================
save_path = f"models/finetuned/{args.type}_finetuned_{args.run_id}.ckpt"

# Because the model was loaded via PyTorch Lightning, we extract its state_dict 
# and save it natively so it can be loaded later.
model.cpu()
torch.save({'state_dict': model.state_dict()}, save_path)
print(f"Training Complete! New model successfully saved to: {save_path}")
