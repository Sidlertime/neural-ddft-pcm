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
dx = float(box_length / domain_resolution)

def get_alpha(t, t_range):
    """
    Smoothly decays the NN multiplier from a starting boost to a final reduction.
    """
    alpha_start = 1.3  # 130% strength at t=0
    alpha_end = 0.6    # 60% strength at the end
    
    # tau controls how fast it decays (t_range/4 means it decays mostly in the first half)
    tau = t_range / 4.0 
    
    #return alpha_end + (alpha_start - alpha_end) * np.exp(-t / tau) # exponential decay
    return alpha_start + (alpha_end - alpha_start)*(t/t_range) + np.sin(t/t_range *2*np.pi)*(1- t/t_range) # sin with linear decaying amplitude (1 to 0) + linear interpolation from alpha_start to alpha_end

def step(rho, v, dt, gamma, eta, m, k, T, V_ext, alpha=1):
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
    flux_centered = rho_faces * v

    # Calculate dynamic Artificial Diffusion to kill the wobble (Upwind stabilization)
    # This precisely targets steep gradients and shocks.
    # Note: Ensure 'dx' is accessible here (e.g., passed as an argument or defined globally)
    grad_rho = field.spatial_gradient(rho)       # Lives on the faces (Staggered)
    D_art = 0.5 * dx * math.abs(v)               # Diffusion coeff based on local velocity
    flux_diffusive = - D_art * grad_rho          # Diffusive flux flows down the gradient
    
    # 3. Combine fluxes for a stable, mass-conserving Upwind Flux
    total_flux = flux_centered + flux_diffusive
    
    # field.divergence calculates ∇(flux) and returns it to the cell centers
    rho_next = rho - dt * field.divergence(total_flux)
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
    dFdrho_grid = dft_U_phiflow(rho_next) * alpha
    exc_term = - (1.0 / m) * field.spatial_gradient(dFdrho_grid)
    #exc_term = - (k * T / m) * field.spatial_gradient(dFdrho_grid)
    #exc_term = 0
    
    # Combine all explicit forces to update velocity
    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + exc_term)
    
    return rho_next, v_next

# Define your time parameters
t_range = 0.1#4.0
#dt = args.dt
save_time_interval = 0.01  # This replaces storage.tracker(1e-2)
#save_step_interval = max(1, int(save_time_interval / dt))
#total_steps = int(t_range / dt)

CFL = 0.4  # Fluid can only move 40% of a cell per step
dt_max = args.dt

V_ext_np = V_ext.values.numpy('x')
current_time = 0.0
target_save_time = save_time_interval
step_num = 0

# Check boundary conditions
print("Density boundaries:", rho.extrapolation)
print("Velocity boundaries:", v.extrapolation)

print("Starting PhiFlow DDFT Simulation with Dynamic Timestepping...")

#for step_num in tqdm(range(total_steps + 1), desc="DDFT Simulation", unit="step", mininterval=1.0):
    # --- SAVING BLOCK ---
    #if step_num % save_step_interval == 0:
        # 1. Extract raw NumPy arrays from the PhiFlow grids
        # (If you get gradient tracking errors, use: rho.values.native().detach().cpu().numpy())
        #rho_data = np.squeeze(rho.values.numpy())
        #v_data = np.squeeze(v.values.vector['x'].numpy())
        
        # 2. Calculate dFdrho using our PhiFlow wrapper, then extract its NumPy array
        #dFdrho_grid = dft_U_phiflow(rho)
        #dFdrho_data = np.squeeze(dFdrho_grid.values.numpy())
        
        # 3. Save the arrays to disk
        #np.save(f"{results_path}/rho{current_time:06.3f}.npy", rho_data)
        #np.save(f"{results_path}/v{current_time:06.3f}.npy", v_data)
        #np.save(f"{results_path}/V_ext{current_time:06.3f}.npy", V_ext_np)
        #np.save(f"{results_path}/dFdrho{current_time:06.3f}.npy", dFdrho_data)
        
        #tqdm.write(f"Saved state at t = {current_time:06.3f}")

    # --- PHYSICS BLOCK ---
    # Advance the simulation by one dt using the step function we defined earlier
    #rho, v = step(rho, v, dt, gamma, eta, m, k, T, V_ext)
    
    # Update time
    #current_time += dt
    
    # Example: Simple steady-state check every 100 steps
    #if step_num % 100 == 0:
        # math.max and math.abs work directly on PhiFlow grids
        #max_vel = math.max(math.abs(v))
        #print(f"Step {step_num}, Max Velocity: {max_vel}")
        
        #if max_vel < 1e-9 and step_num > 1000:
        #    print("Steady state reached!")
        #    break
with tqdm(total=t_range, desc="DDFT Simulation", unit="sim_time", 
          bar_format="{l_bar}{bar}| [{elapsed}<{remaining} {postfix}]") as pbar:
    
    while current_time < t_range:
        # --- 1. CALCULATE DYNAMIC dt ---
        v_raw = v.values.vector['x'].numpy()
        v_max = float(np.max(np.abs(v_raw)))
        
        # KILL SWITCH: Stop if the simulation blows up
        if np.isnan(v_max) or np.isinf(v_max):
            tqdm.write("ABORT: Velocity exploded to NaN or Infinity!")
            break
        
        dt = CFL * dx / (v_max + 1e-6)
        dt = min(dt, dt_max)
        
        # Dont overshoot end of simulation
        dt = min(dt, t_range - current_time)

        # --- UPDATE TQDM POSTFIX ---
        pbar.set_postfix(
            time=f"{current_time:.4f}/{t_range:.4f}", 
            dt=f"{dt:.2e}", 
            vmax=f"{v_max:.4f}"
        )

        # --- PHYSICS BLOCK ---
        current_alpha = get_alpha(current_time, t_range) # boost or nerf the dFdrho
        rho, v = step(rho, v, dt, gamma, eta, m, k, T, V_ext, current_alpha)
        current_time += dt
        step_num += 1
        
        # Update progress bar
        pbar.update(dt)

        # --- SAVING BLOCK ---
        if current_time >= target_save_time or current_time >= t_range-1e-8:
            rho_data = np.squeeze(rho.values.numpy())
            v_data = np.squeeze(v.values.vector['x'].numpy())
            
            dFdrho_grid = dft_U_phiflow(rho)
            dFdrho_data = np.squeeze(dFdrho_grid.values.numpy())
            dFdrho_mod_grid = dFdrho_grid * current_alpha
            dFdrho_mod_data = np.squeeze(dFdrho_mod_grid.values.numpy())
            
            np.save(f"{results_path}/rho{current_time:06.3f}.npy", rho_data)
            np.save(f"{results_path}/v{current_time:06.3f}.npy", v_data)
            np.save(f"{results_path}/V_ext{current_time:06.3f}.npy", V_ext_np)
            np.save(f"{results_path}/dFdrho{current_time:06.3f}.npy", dFdrho_data)
            np.save(f"{results_path}/dFdrho_mod{current_time:06.3f}.npy", dFdrho_mod_data)

            # Increment the threshold for the next save
            target_save_time += save_time_interval
            
            # Optional: comment this out if the progress bar updates are enough for you!
            # tqdm.write(f"Saved state at t = {current_time:06.3f}")
    

print("Simulation complete! Generating movie...")
plot_movie(results_path, args.md_path, args.ref_md_path, True)
