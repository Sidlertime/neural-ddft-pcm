import os
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
from tqdm import tqdm
from phi.torch.flow import *

from neural_ddft.utils.config import load_config, get_next_result_path, save_config
from neural_ddft.utils.functions import plot_movie
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.models.zimmermann_net import ZimmermannNet
from neural_ddft.models.neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT

PARAM_SCHEME = {
    "run_id": (str, "phiflow_eval"),
    "md_path": (str, "../data/g1-closed-dz100/avg_5000"),
    "ref_md_path": (str, None),
    "model_mode": (str, "hybrid"),              # "equilibrium", "hybrid", or "zimmermann"
    "base_model_path": (str, "models/c2_dz32.ckpt"),
    "sup_model_path": (str, None),
    "zimmermann_model_path": (str, None),
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 2.0),
    "eta": (float, 0.01),
    "dt": (float, 1e-4),
}

if __name__ == "__main__":
    # 1. Load Config & Setup Paths
    args = load_config(PARAM_SCHEME, description="Modular Neural DDFT")
    
    current_time = datetime.now().strftime("%Y-%m-%d")
    base_path = f"results/{current_time}_{args['run_id']}"
    results_path = get_next_result_path(base_path)
    os.makedirs(results_path, exist_ok=True)
    save_config(args, results_path) # Permanently record these settings
    
    # 2. Load MD Data Bounds
    md_files = sorted(glob.glob(f"{args['md_path']}/collection.*.h5"))
    with h5py.File(md_files[0], 'r') as f:
        init_rho = np.maximum(np.array(f['n']).flatten(), 1e-6)
        V = np.array(f['V']).flatten()
        r = np.array(f['r']).flatten()

    box_length = float(r[-1] - r[0])
    domain_res = r.shape[0]
    dx = box_length / domain_res
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load Models
    models = {}

    if int(1/dx) == 100:
        models["base"] = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=args["base_model_path"], map_location=device)
    else:
        models["base"] = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=args["base_model_path"], map_location=device)
    models["base"].eval()

    if args['model_mode'] == "hybrid":
        models['sup'] = SuperadiabaticNet(channels=16).to(device)
        if args['sup_model_path']:
            models['sup'].load_state_dict(torch.load(args["sup_model_path"], map_location=device))
        models['sup'].eval()
    elif args['model_mode'] == "zimmermann":
        models["zimmermann"] = ZimmermannNet().to(device)
        if args['zimmermann_model_path']:
            models["zimmermann"].load_state_dict(torch.load(args['zimmermann_model_path'], map_location=device))
        models["zimmermann"].eval()

    # 4. Initialize PhiFlow Grids & Solver
    rho = CenteredGrid(math.tensor(init_rho, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.tensor(V, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    solver = PhiFlowDDFT(
        models=models, device=device, dx=dx, config=args # Pass the whole config to access gamma, eta, k, T, m inside the step function
    )

    # 5. The Main Physics Loop
    t_range, save_interval, CFL, current_time, target_save = 4.0, 0.01, 0.4, 0.0, 0.01
    
    with tqdm(total=t_range, desc="DDFT Simulation", unit="sim_time", bar_format="{l_bar}{bar}| [{elapsed}<{remaining} {postfix}]") as pbar:
        while current_time < t_range:
            v_max = float(np.max(np.abs(v.values.vector['x'].native('batch,x').cpu().detach().numpy())))
            if np.isnan(v_max) or np.isinf(v_max): break
            
            dt = min(CFL * dx / (v_max + 1e-6), args['dt'], t_range - current_time)
            
            rho, v = solver.step(rho, v, dt, V_ext)
            
            current_time += dt
            pbar.update(dt)
            pbar.set_postfix(dt=f"{dt:.2e}", vmax=f"{v_max:.4f}")

            # 6. Save State
            if current_time >= target_save or current_time >= t_range - 1e-8:
                solver.save_state(rho, v, V_ext, current_time, results_path)
                target_save += save_interval

    print("Simulation complete! Generating movie...")
    plot_movie(results_path, args['md_path'], args['ref_md_path'], True)
