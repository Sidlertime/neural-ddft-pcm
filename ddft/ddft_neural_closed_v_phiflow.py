import pde
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

from neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from functions import plot_movie

from phi.torch.flow import *
import matplotlib.pyplot as plt
from tqdm import tqdm

#################
##### Setup #####
#################

parser = argparse.ArgumentParser(description='neural DDFT')
parser.add_argument('--run_id', type=str, default='g1-closed')
parser.add_argument('--md_path', type=str, default='../data/g1-closed-dz100/avg_5000')
parser.add_argument('--ref_md_path', type=str, default=None)
parser.add_argument('--type', type=str, default='c2')
parser.add_argument('--k', type=float, default=1)
parser.add_argument('--T', type=float, default=2)
parser.add_argument('--m', type=float, default=1)
parser.add_argument('--gamma', type=float, default=2)
parser.add_argument('--eta', type=float, default=0.01)
parser.add_argument('--dt', type=float, default=1e-4)

args = parser.parse_args()

current_time = datetime.now().strftime("%Y-%m-%d")
results_path = f"results/{current_time}_{args.run_id}_k{args.k}_T{args.T}_m{args.m}_g{args.gamma}_eta{args.eta}_dt{args.dt}_{args.type}"
os.makedirs(results_path, exist_ok=True)

# Load MD data

file_pattern = f"{args.md_path}/collection.*.h5"
md_file_list = sorted(glob.glob(file_pattern))

if len(md_file_list) == 0:
    raise FileNotFoundError(f"No MD files found matching pattern: {file_pattern}")

with h5py.File(md_file_list[0], 'r') as f:
    init_rho = np.array(f['n']).flatten()
    V = np.array(f['V']).flatten()
    r = np.array(f['r']).flatten()

dz = r[1] - r[0]
print(f'Number of particles: {np.sum(init_rho) * dz * 10 * 10} (dz: {dz})')

# Load F model
if int(1/dz) == 100:
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=f"models/{args.type}_dz100.ckpt", map_location=torch.device('cpu'))
else:
    model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=f"models/{args.type}_dz{int(1/dz)}.ckpt", map_location=torch.device('cpu'))
    
model.eval()

# Define Domain and Grids
# We define a 1D periodic box.
box_length = r[-1] - r[0]
domain_resolution = r.shape[0]

# rho must not be exactly zero to avoid division errors in the pressure/viscosity terms
init_rho_safe = np.maximum(init_rho, 1e-6)

# Initialize Fields
# Density (rho) and External Potential (V_ext) live in the cell centers
rho_tensor = math.tensor(init_rho_safe, spatial('x'))
V_tensor = math.tensor(V, spatial('x'))
rho = CenteredGrid(rho_tensor, extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))
V_ext = CenteredGrid(V_tensor, extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))

# Velocity (v) lives on the cell faces (StaggeredGrid) to prevent instability
v_tensor = math.zeros(spatial(x=domain_resolution))
v = StaggeredGrid(v_tensor, extrapolation.PERIODIC, x=domain_resolution, bounds=Box(x=box_length))

# Parameters
dt = args.dt
k = args.k
T = args.T
gamma = args.gamma   # Friction
m = args.m     # Particle mass
eta = args.eta   # Viscosity (set to 0 if you want pure Langevin DDFT without hydrodynamics)

#################
##### Model #####
#################

# Define free energy potential --- with gradient operation
def dft_U_phiflow(rho_grid):
    """
    Wraps the PyTorch NN to compute δF_exc/δρ natively within PhiFlow.
    Expects a CenteredGrid and returns a CenteredGrid.
    """
    # 1. Extract raw PyTorch tensor and reshape to (B, C, N) -> (1, 1, N)
    rho_pt = rho_grid.values.native()
    rho_tensor = rho_pt.view(1, 1, -1).float().requires_grad_(True)
    
    # PhiFlow grids know their spacing. We extract dz dynamically:
    dz = float(np.squeeze(rho_grid.dx.numpy()))
    
    # 2. Run through Neural Network
    torch.set_grad_enabled(True)
    F_pred = model(rho_tensor)
    # This computes the derivative δF_exc/δρ
    dFdrho = torch.autograd.grad(F_pred, rho_tensor)[0] / dz
    torch.set_grad_enabled(False)
    
    # 3. Squeeze back to 1D: shape (N,)
    dFdrho_1d = dFdrho.squeeze()
    
    # 4. Repackage into a PhiFlow CenteredGrid
    dFdrho_phi = math.tensor(dFdrho_1d, spatial('x'))
    return CenteredGrid(dFdrho_phi, extrapolation.PERIODIC, bounds=rho_grid.bounds)

################
##### DDFT #####
################

def step(rho, v, dt, gamma, eta, m, k, T, V_ext):
    """
    Performs one integration step of the DDFT equations.
    rho: CenteredGrid
    v: StaggeredGrid
    """
    # ==========================================
    # EQUATION 1: ∂ρ/∂t = -∇(ρv)
    # ==========================================
    # Use rho.at() to interpolate density to the velocity grid faces
    rho_faces = rho.at(v)
    flux = rho_faces * v
    
    # field.divergence calculates ∇(flux) and returns it to the cell centers
    rho_next = rho - dt * field.divergence(flux)
    rho_next = rho_next.with_values(math.maximum(rho_next.values, 1e-6))
    
    # ==========================================
    # EQUATION 2: ∂v/∂t = ... (Velocity Balance)
    # ==========================================
    
    # a) Advection: -v∇v 
    v_advected = advect.semi_lagrangian(v, v, dt)
    
    # Resample updated density to faces for the remaining division terms
    rho_faces_next = rho_next.at(v)
    
    # b) Friction: -γv
    fric_term = -gamma * v
    
    # c) Viscous Diffusion: + (η/(mρ)) ∇²v
    # Use field.laplace for physical grids
    visc_term = (eta / (m * rho_faces_next)) * field.laplace(v)
    
    # d) Ideal Gas Pressure: - (k_B T / (mρ)) ∇ρ
    # Use field.spatial_gradient to map from Centers to Faces
    pressure_term = - (k * T / (m * rho_faces_next)) * field.spatial_gradient(rho_next)
    
    # e) External Potential: - (1/m) ∇V_ext
    vext_term = - (1.0 / m) * field.spatial_gradient(V_ext)
    
    # f) Excess Free Energy: - (1/m) ∇(δF_exc/δρ)
    dFdrho_grid = dft_U_phiflow(rho_next)
    exc_term = - (1.0 / m) * field.spatial_gradient(dFdrho_grid)
    #exc_term = - (k * T / m) * field.spatial_gradient(dFdrho_grid)
    
    # Combine all explicit forces to update velocity
    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + exc_term)
    
    return rho_next, v_next

# Define your time parameters
t_range = 4.0
dt = args.dt
save_time_interval = 0.01  # This replaces storage.tracker(1e-2)
save_step_interval = max(1, int(save_time_interval / dt))
total_steps = int(t_range / dt)

V_ext_np = V_ext.values.numpy('x')
current_time = 0.0

for step_num in tqdm(range(total_steps + 1), desc="DDFT Simulation", unit="step", mininterval=1.0):
    
    # --- SAVING BLOCK ---
    if step_num % save_step_interval == 0:
        # 1. Extract raw NumPy arrays from the PhiFlow grids
        # (If you get gradient tracking errors, use: rho.values.native().detach().cpu().numpy())
        rho_data = np.squeeze(rho.values.numpy())
        v_data = np.squeeze(v.values.vector['x'].numpy())
        
        # 2. Calculate dFdrho using our PhiFlow wrapper, then extract its NumPy array
        dFdrho_grid = dft_U_phiflow(rho)
        dFdrho_data = np.squeeze(dFdrho_grid.values.numpy())
        
        # 3. Save the arrays to disk
        np.save(f"{results_path}/rho{current_time:06.3f}.npy", rho_data)
        np.save(f"{results_path}/v{current_time:06.3f}.npy", v_data)
        np.save(f"{results_path}/V_ext{current_time:06.3f}.npy", V_ext_np)
        np.save(f"{results_path}/dFdrho{current_time:06.3f}.npy", dFdrho_data)
        
        #print(f"Saved state at t = {current_time:06.3f}")

    # --- PHYSICS BLOCK ---
    # Advance the simulation by one dt using the step function we defined earlier
    rho, v = step(rho, v, dt, gamma, eta, m, k, T, V_ext)
    
    # Update time
    current_time += dt
    
    # Example: Simple steady-state check every 100 steps
    #if step_num % 100 == 0:
        # math.max and math.abs work directly on PhiFlow grids
        #max_vel = math.max(math.abs(v))
        #print(f"Step {step_num}, Max Velocity: {max_vel}")
        
        #if max_vel < 1e-9 and step_num > 1000:
        #    print("Steady state reached!")
        #    break

print("Simulation complete! Generating movie...")
plot_movie(results_path, args.md_path, args.ref_md_path)
