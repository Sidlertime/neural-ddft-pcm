import pde
import numpy as np
import h5py
import glob
from datetime import datetime
import os
import argparse

from neural_ddft.utils.functions import plot_movie_closed
from neural_ddft.physics.fmt import fmt_module

##################
##### Setup #####
##################

parser = argparse.ArgumentParser(description='fmt DDFT')
parser.add_argument('--run_id', type=str, default='test-fmt')
parser.add_argument('--md_path', type=str, default='../data/g1-closed-dz100/avg_5000')

args = parser.parse_args()

# Load F model
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

################
##### DDFT #####
################

model = fmt_module.FMTFreeEnergy(dz=dz)

# Define free energy potential
def dft_U(rho):
    rho = np.expand_dims(np.expand_dims(rho, 0), 0).astype(np.float32)
    dFdrho = model(rho, 0.60)
    dFdrho = np.squeeze(np.squeeze(dFdrho))
    return dFdrho

# Define grid, field
grid = pde.CartesianGrid([[r[0], r[-1]]], r.shape[0], periodic=True)
rho = pde.ScalarField(grid, data=init_rho)
V_ext_field = pde.ScalarField(grid, data=V)

k = 1
T = 2
gamma = 8
lap_rho = "((k*T) / gamma) * laplace(rho)"
div_V = "(1 / gamma) * divergence(rho * gradient(V_ext_field))"
div_F = "((k*T) / gamma) * divergence(rho * gradient(dft_u(rho)))"

eq = pde.PDE({"rho": f"{lap_rho} + {div_V} + {div_F}"},
            user_funcs={"dft_u": dft_U},
            consts={"V_ext_field": V_ext_field, "T": T, "k": k, "gamma": gamma},
            bc=[{"type": "periodic"}])
storage = pde.MemoryStorage()

# Solve
result = eq.solve(rho, t_range=4, dt=1e-4,
                  tracker=["progress", storage.tracker(1e-2), pde.SteadyStateTracker()])

# Save results
ddft_rho_list = []
for i, field in storage.items():
    # Save density field
    np.save(f"{results_path}/rho{i:06.3f}.npy", field.data)
    
    # Save external potential
    np.save(f"{results_path}/V_ext{i:06.3f}.npy", V_ext_field.data)
    
    # Calculate and save dFdrho
    dFdrho = dft_U(field.data)
    np.save(f"{results_path}/dFdrho{i:06.3f}.npy", dFdrho)

np.save(f"{results_path}/rho_init.npy", init_rho)

# Print final number of particles
print('Number of ddft particles: ', np.sum(field.data) * dz * 10 * 10)
with h5py.File(md_file_list[-1], 'r') as f:
    n = np.array(f['n']).flatten()
    print('Number of MD particles: ', np.sum(n) * dz * 10 * 10)

plot_movie_closed(results_path, args.md_path)