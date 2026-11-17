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
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.utils.functions import plot_movie
from neural_ddft.utils.config import load_config, get_next_result_path, save_config

##################
##### Setup #####
##################

PARAM_SCHEME = {
    "run_id": (str, "pypde_eval"),
    "md_path": (str, "../data/lj_data/ref_3a_langevin_01/avg_2000/"),
    "ref_md_path": (str, None),
    "type": (str, "c2"),
    #"base_model": (str, None),
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 2.0),
    "eta": (float, 0.01),
    "dt": (float, 1e-4),
    # Optional learned superadiabatic correction (hybrid mode) - checkpoint
    # path plus the architecture flags it was trained with (must match
    # exactly, same requirement as neural-ddft-pcm's own checkpoint-
    # consistency check in factory.py).
    "sup_model_path": (str, None),
    "sup_use_grad": (bool, True),
    "sup_use_vext": (bool, False),
    "sup_use_rho_prev": (bool, False),
    "sup_use_v_prev": (bool, False),
    "sup_use_accel": (bool, False),
    "sup_use_gamma": (bool, False),
    "sup_use_eta": (bool, False),
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

sup_model = None
if args.get('sup_model_path'):
    sup_model = SuperadiabaticNet(
        channels=16, use_grad=args['sup_use_grad'], use_vext=args['sup_use_vext'],
        use_rho_prev=args['sup_use_rho_prev'], use_v_prev=args['sup_use_v_prev'],
        use_accel=args['sup_use_accel'], use_gamma=args['sup_use_gamma'], use_eta=args['sup_use_eta'],
    )
    sup_ckpt = torch.load(args['sup_model_path'], map_location=torch.device('cpu'))
    sup_state_dict = sup_ckpt['state_dict'] if isinstance(sup_ckpt, dict) and 'state_dict' in sup_ckpt else sup_ckpt
    sup_model.load_state_dict(sup_state_dict)
    sup_model.eval()
    print(f"Loaded hybrid correction network from {args['sup_model_path']}")

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


# Simple same-object cache for rho_prev/v_prev - py-pde's string-expression
# PDE system calls sup_f as a stateless function at arbitrary (possibly
# out-of-order, due to adaptive substepping) t, so "previous step" has no
# precise meaning here; same None->self fallback convention as
# force_factory.py's own rho_prev_grid=None case.
_prev_cache = {"rho": None, "v": None}


def sup_f(rho, v):
    """Optional learned superadiabatic correction force, same channel-
    building convention as neural-ddft-pcm's force_factory.py:get_hybrid_force."""
    if sup_model is None:
        return np.zeros_like(rho)

    rho_t = torch.from_numpy(rho).view(1, -1).float()
    v_t = torch.from_numpy(v).view(1, -1).float()
    grad_rho_t = (torch.roll(rho_t, -1, dims=-1) - torch.roll(rho_t, 1, dims=-1)) / (2 * dz)
    grad_v_t = (torch.roll(v_t, -1, dims=-1) - torch.roll(v_t, 1, dims=-1)) / (2 * dz)

    extra_kwargs = {}
    if sup_model.use_vext:
        extra_kwargs["v_ext"] = torch.from_numpy(V).view(1, -1).float()
    needs_rho_prev = sup_model.use_rho_prev
    needs_v_prev = sup_model.use_v_prev
    needs_accel = sup_model.use_accel
    if needs_rho_prev or needs_accel:
        prev = _prev_cache["rho"] if _prev_cache["rho"] is not None else rho
        rho_prev_t = torch.from_numpy(prev).view(1, -1).float()
        if needs_rho_prev:
            extra_kwargs["rho_prev"] = rho_prev_t
    if needs_v_prev or needs_accel:
        prev = _prev_cache["v"] if _prev_cache["v"] is not None else v
        v_prev_t = torch.from_numpy(prev).view(1, -1).float()
        if needs_v_prev:
            extra_kwargs["v_prev"] = v_prev_t
    if needs_accel:
        extra_kwargs["accel"] = (v_t - v_prev_t) / args['dt']
    if sup_model.use_gamma:
        extra_kwargs["gamma"] = torch.full_like(rho_t, float(args['gamma']))
    if sup_model.use_eta:
        extra_kwargs["eta"] = torch.full_like(rho_t, float(args['eta']))

    with torch.no_grad():
        f_sup = sup_model(rho_t, v_t, grad_rho_t, grad_v_t, **extra_kwargs).squeeze(0).numpy()

    _prev_cache["rho"], _prev_cache["v"] = rho.copy(), v.copy()
    return f_sup

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

# Optional learned superadiabatic correction (hybrid mode, see sup_f above) -
# zero when no sup_model_path was given, so this is a no-op extension of
# the original equation in that case.
term_sup = "+ (1/m) * sup_f(rho, v)"

# Combine
eq_v = f"{term_advection} {term_friction} {term_force} {term_viscosity} {term_sup}"

# Combine into PDE system
eq = pde.PDE(
    {"rho": eq_rho, "v": eq_v},
    user_funcs={"dft_u": dft_U, "safe_log": safe_log, "upwind_adv": upwind_adv, "sup_f": sup_f},
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