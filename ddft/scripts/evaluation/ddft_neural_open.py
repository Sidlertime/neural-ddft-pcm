import pde
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

from neural_ddft.neural_free_energy import NeuralFreeEnergy
from neural_ddft.functions import plot_movie


##################
##### Setup #####
##################

parser = argparse.ArgumentParser(description='gcmc DDFT')
parser.add_argument('--run_id', type=str, default='test-open')
parser.add_argument('--md_path', type=str, default='../data/irmof10-open-dz32/avg_500')
args = parser.parse_args()

# Load F model
current_time = datetime.now().strftime("%Y-%m-%d")
results_path = f"results/{current_time}-{args.run_id}"
os.makedirs(results_path, exist_ok=True)

model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path="models/c2_dz32.ckpt", map_location=torch.device('cpu'))
model.eval()

# Load MD data
file_pattern = f"{args.md_path}/collection.*.h5"
md_file_list = sorted(glob.glob(file_pattern))

if len(md_file_list) == 0:
    raise FileNotFoundError(f"No MD files found matching pattern: {file_pattern}")

with h5py.File(md_file_list[0], 'r') as f:
    init_rho = np.array(f['n']).flatten()
    # replace zeroes with small number
    init_rho[init_rho == 0] = 1e-4
    V = np.array(f['V']).flatten()
    r = np.array(f['r']).flatten()

dz = r[1] - r[0]
print('Number of particles: ', np.sum(init_rho) * dz * 10 * 10)

################
##### DDFT #####
################

# Define free energy potential
def dft_U(open_rho):
    rho = torch.from_numpy(open_rho).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
    torch.set_grad_enabled(True)
    F_pred = model(rho)
    dFdrho = torch.autograd.grad(F_pred, rho)[0] / dz
    torch.set_grad_enabled(False)
    dFdrho = dFdrho.squeeze().squeeze().detach().numpy()
    return dFdrho

# Define grid, field
grid = pde.CartesianGrid([[r[0], r[-1]]], r.shape[0])
rho = pde.ScalarField(grid, data=init_rho)
V_ext_field = pde.ScalarField(grid, data=V)

# Define the transition points and epsilon for the smoothness
r1 = 2
r2 = 8
mask1 = pde.ScalarField(grid, data = 1)
mask1.data[r < r1] = 0

mask2 = pde.ScalarField(grid, data = 1)
mask2.data[r > r2] = 0

k = 1
T = 2
gamma = 10
mu1 = 0.5
mu2 = -100

# Define equation
growth1 = "(1 - mask1) * rho * (mu1 - V_ext_field - (k*T) * ln(rho) - (k*T) * dft_u(rho))"
growth2 = "(1 - mask2) * rho * (mu2 - V_ext_field - (k*T) * ln(rho) - (k*T) * dft_u(rho))"

lap_rho = "((k*T) / gamma) * laplace(rho)"
div_V = "(1 / gamma) * divergence(rho * gradient(V_ext_field))"
div_F = "((k*T) / gamma) * divergence(rho * gradient(dft_u(rho)))"

eq = pde.PDE({"rho": f"{lap_rho} + {div_V} + {div_F} + {growth1} + {growth2}"},
            user_funcs={"dft_u": dft_U},
            consts={"V_ext_field": V_ext_field, "mask1": mask1, "mask2": mask2, "T": T, "k": k, "mu1": mu1, "mu2": mu2, "gamma": gamma})
storage = pde.MemoryStorage()

# Solve
try:
    result = eq.solve(rho, t_range=150, dt=1e-3,
                      tracker=["progress", storage.tracker(1e-1)])
except Exception as e:
    print(f"Exception occurred: {e}")
    result = None

# Save results
ddft_rho_list = []
for i, field in storage.items():
    # Save density field
    np.save(f"{results_path}/rho{i:07.3f}.npy", field.data)
    
    # Save external potential
    np.save(f"{results_path}/V_ext{i:07.3f}.npy", V_ext_field.data)
    
    # Calculate and save dFdrho
    dFdrho = dft_U(field.data)
    np.save(f"{results_path}/dFdrho{i:07.3f}.npy", dFdrho)

# Print final number of particles
print('Number of ddft particles: ', np.sum(field.data) * dz * 10 * 10)
with h5py.File(md_file_list[-1], 'r') as f:
    n = np.array(f['n']).flatten()
    print('Number of MD particles: ', np.sum(n) * dz * 10 * 10)

plot_movie(results_path, args.md_path)