import pde
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

from neural_free_energy import NeuralFreeEnergyFine
from functions import plot_movie

##################
##### Setup #####
##################

parser = argparse.ArgumentParser(description='neural DDFT')
parser.add_argument('--run_id', type=str, default='g1-closed')
parser.add_argument('--md_path', type=str, default='../data/g1-closed-dz100/avg_5000')
parser.add_argument('--type', type=str, default='c2')

args = parser.parse_args()

current_time = datetime.now().strftime("%Y-%m-%d")
results_path = f"results/{current_time}-{args.run_id}"
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
print('Number of particles: ', np.sum(init_rho) * dz * 10 * 10)

# Load F model
if args.type == 'c2':
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path="models/c2_dz100.ckpt", map_location=torch.device('cpu'))

elif args.type == 'c1':
    model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path="models/c1_dz100.ckpt", map_location=torch.device('cpu'))
    
model.eval()

################
##### DDFT #####
################

# Define free energy potential
def dft_U(rho):
    rho = torch.from_numpy(rho).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
    torch.set_grad_enabled(True)
    F_pred = model(rho)
    dFdrho = torch.autograd.grad(F_pred, rho)[0] / dz
    torch.set_grad_enabled(False)
    dFdrho = dFdrho.squeeze().squeeze().detach().numpy()
    return dFdrho

# Define grid, field
grid = pde.CartesianGrid([[r[0], r[-1]]], r.shape[0], periodic=True)
rho_field = pde.ScalarField(grid, data=init_rho)
V_ext_field = pde.ScalarField(grid, data=V)

# Initialize State: We need a Collection of [rho, j]
j_field = pde.VectorField(grid)     # j (momentum density) starts at 0
state = pde.FieldCollection([rho_field, j_field])

# Parameters
k = 1
T = 2
gamma = 8

# Equations
# Continuity: dt(rho) = -div(j)
eq_rho = "-divergence(j)"

# Momentum: dt(j) = Forces - Friction
# Force Ideal = -k*T * grad(rho)  (Ideal Gas Pressure gradient)
# Force Ext   = -rho * grad(V)
# Force Ex    = -rho * grad(mu_ex)
# Note: In your previous code, dft_u had a k*T multiplier, V did not. 
# I preserved that scaling here.
eq_j = ("- k*T * gradient(rho) "
        "- rho * gradient(V_ext_field) "
        "- (k*T) * rho * gradient(dft_u(rho)) "
        "- gamma * j")

# Combine into PDE system
eq = pde.PDE(
    {"rho": eq_rho, "j": eq_j},
    user_funcs={"dft_u": dft_U},
    consts={"V_ext_field": V_ext_field, "T": T, "k": k, "gamma": gamma},
    bc=[{"type": "periodic"}, {"type": "periodic"}] # BCs for rho and j
)
storage = pde.MemoryStorage()

# Solve
# Note: dt usually needs to be smaller for inertial dynamics to remain stable
result = eq.solve(state, t_range=4, dt=1e-5,
                  tracker=["progress", storage.tracker(1e-2)],
                  backend='numpy')

# Save results
for i, field_collection in storage.items():
    rho_data = field_collection[0].data  # Extract rho
    j_data = field_collection[1].data    # Extract j
    
    # Save density field
    np.save(f"{results_path}/rho{i:06.3f}.npy", rho_data)
    
    # Save momentum field (optional, but useful for checking convergence)
    np.save(f"{results_path}/j{i:06.3f}.npy", j_data)
    
    # Save external potential
    np.save(f"{results_path}/V_ext{i:06.3f}.npy", V_ext_field.data)
    
    # Calculate and save dFdrho
    dFdrho = dft_U(rho_data)
    np.save(f"{results_path}/dFdrho{i:06.3f}.npy", dFdrho)

plot_movie(results_path, args.md_path)