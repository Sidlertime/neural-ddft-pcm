import pde
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb

from neural_ddft.models.neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from neural_ddft.utils.functions import plot_movie
from neural_ddft.utils.config import load_config, get_next_result_path, save_config

##################
##### Setup #####
##################

PARAM_SCHEME = {
    "run_id": (str, "pypde_eval"),
    "md_path": (str, "../data/g1-closed-dz100/avg_5000"),
    "ref_md_path": (str, None),
    "type": (str, "c2"),
    #"base_model": (str, None),
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 2.0),
    "eta": (float, 0.01),
    "dt": (float, 1e-4),
}

args = load_config(PARAM_SCHEME, description="Modular Neural DDFT in PyPDE")

current_time = datetime.now().strftime("%Y-%m-%d")
base_path = f"results/{current_time}_{args['run_id']}"
results_path = get_next_result_path(base_path)
os.makedirs(results_path, exist_ok=True)
save_config(args, results_path)

# Load MD data

file_pattern = f"{args['md_path']}/collection.*.h5"
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
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=f"models/{args['type']}_dz100.ckpt", map_location=torch.device('cpu'))
else:
    model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=f"models/{args['type']}_dz{int(1/dz)}.ckpt", map_location=torch.device('cpu'))
    
model.eval()

################
##### DDFT #####
################

# Define free energy potential --- with gradient operation
def dft_U(rho, smoothing=True):
    rho = torch.from_numpy(rho).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
    torch.set_grad_enabled(True)
    F_pred = model(rho)
    dFdrho = torch.autograd.grad(F_pred, rho)[0] / dz
    torch.set_grad_enabled(False)
    dFdrho = dFdrho.squeeze().squeeze().detach().numpy()
    # Apply a mild filter to prevent PDE explosions
    # You can tune sigma. sigma=1.0 or 2.0 is usually a good starting point.
    if smoothing:
        dFdrho = gaussian_filter1d(dFdrho, sigma=2.0)
    return dFdrho

# rho must not be exactly zero to avoid division errors in the pressure/viscosity terms
init_rho_safe = np.maximum(init_rho, 1e-6)

# Perform a Dry Run and Plot
raw_grad = dft_U(init_rho_safe, False)
smooth_grad = dft_U(init_rho_safe)

plt.figure(figsize=(8, 4))
# Plotting the smooth version
plt.plot(raw_grad, label="Raw NN Gradient", color='black')
plt.plot(smooth_grad, label="Smoothed NN Gradient (Going to Solver)", color='red')
plt.title("Initial $\\delta F_{exc} / \\delta \\rho$ check")
plt.legend()
plt.savefig(f"{results_path}/_nn_gradient_check.png", bbox_inches="tight")

# Define grid, field
print(f"Grid Points: {r.shape[0]}")
print(f"Grid Range: {r[0]} - {r[-1]}")
grid = pde.CartesianGrid([[r[0], r[-1]]], r.shape[0], periodic=True)
rho_field = pde.ScalarField(grid, data=init_rho_safe)
V_ext_field = pde.ScalarField(grid, data=V)

# Initialize State: We need a Collection of [rho, v]
#v_field = pde.VectorField(grid)     # velocity starts at 0
v_field = pde.ScalarField(grid, data=0.0)     # velocity starts at 0
state = pde.FieldCollection([rho_field, v_field])

# Parameters
k = args['k']
T = args['T']
gamma = args['gamma']   # Friction
m = args['m']           # Particle mass
eta = args['eta']       # Viscosity (set to 0 if you want pure Langevin DDFT without hydrodynamics)

def safe_log(rho):
    return np.log(np.maximum(rho, 1e-10))

def upwind_adv(v):
    # Extract dx directly from your instantiated grid
    dx = grid.discretization[0]
    
    # Pad the array for periodic boundary conditions ('wrap' = periodic)
    v_pad = np.pad(v, 1, mode='wrap')
    
    # Calculate differences
    v_dx_forward = (v_pad[2:] - v_pad[1:-1]) / dx
    v_dx_backward = (v_pad[1:-1] - v_pad[:-2]) / dx
    
    # Upwind logic: Return a new array containing the advection term
    return np.where(v > 0, v * v_dx_backward, v * v_dx_forward)

# Equations
# Continuity: dt(rho) = -div(rho * v)
#eq_rho = "-divergence(rho * v)"
eq_rho = "-d_dx(rho * v)"

# Velocity Equation terms:
# Advection: -(v . grad) v
#term_advection = "- 0.5 * gradient(dot(v, v))"
term_advection = "- upwind_adv(v)"

# Friction: - gamma * v
term_friction = "- gamma * v"

# Forces: - (1/m) * grad(mu)
# mu = V_ext + mu_excess + k*T*ln(rho)
# grad(mu) = grad(V) + grad(mu_ex) + (kT/rho)*grad(rho)
#term_force = ("- (1/m) * ("
#              "gradient(V_ext_field) + "
#              "(k*T) * gradient(dft_u(rho)) + "
#              "(k*T) * gradient(safe_log(rho))" 
#              ")")
term_force = ("- (1/m) * ("
              "d_dx(V_ext_field) + "
              "(k*T) * d_dx(dft_u(rho)) + "
              "(k*T) * d_dx(safe_log(rho))" 
              ")")

# Viscosity: (eta / m*rho) * laplace(v)
#term_viscosity = "+ (eta / (m * rho)) * vector_laplace(v)"
term_viscosity = "+ (eta / (m * rho)) * laplace(v)"

# Combine
eq_v = f"{term_advection} {term_friction} {term_force} {term_viscosity}"

# Combine into PDE system
eq = pde.PDE(
    {"rho": eq_rho, "v": eq_v},
    user_funcs={"dft_u": dft_U, "safe_log": safe_log, "upwind_adv": upwind_adv},
    consts={"V_ext_field": V_ext_field, "T": T, "k": k, "gamma": gamma, "m": m, "eta": eta},
    bc=[{"type": "periodic"}, {"type": "periodic"}] # BCs for rho and v
)
storage = pde.MemoryStorage()

# Solve
# Note: dt usually needs to be smaller for inertial dynamics to remain stable
result = eq.solve(state, t_range=4, dt=args['dt'],
                  tracker=["progress", storage.tracker(1e-2)],
                  backend='numpy')

# Save results
for i, field_collection in storage.items():
    rho_data = field_collection[0].data  # Extract rho
    v_data = field_collection[1].data    # Extract v
    
    # Save density field
    np.save(f"{results_path}/rho{i:06.3f}.npy", rho_data)
    
    # Save momentum field (optional, but useful for checking convergence)
    np.save(f"{results_path}/v{i:06.3f}.npy", v_data)
    
    # Save external potential
    np.save(f"{results_path}/V_ext{i:06.3f}.npy", V_ext_field.data)
    
    # Calculate and save dFdrho
    dFdrho = dft_U(rho_data)
    np.save(f"{results_path}/dFdrho{i:06.3f}.npy", dFdrho)

plot_movie(results_path, args['md_path'], args['ref_md_path'])