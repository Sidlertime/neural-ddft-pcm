import os
import time
import torch
import numpy as np
import h5py
import glob
from datetime import datetime
from tqdm import tqdm
from phi.torch.flow import *

from neural_ddft.utils.config import load_config, get_next_result_path, save_config
from neural_ddft.utils.functions import plot_movie
from neural_ddft.utils.metrics import summarize_run
from neural_ddft.utils.run_summary import write_run_summary
from neural_ddft.models.factory import build_models
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT

PARAM_SCHEME = {
    "run_id": (str, "phiflow_eval"),
    #"md_path": (str, "../data/g1-closed-dz100/avg_5000"),
    "md_path": (str, "../data/lj_data/ref_gaussian_langevin_01/avg_500/"),
    "ref_md_path": (str, None),
    "model_mode": (str, "hybrid"),              # "equilibrium", "hybrid", or "zimmermann"
    "base_model_path": (str, "models/c2_dz32.ckpt"),
    "sup_model_path": (str, None),
    "zimmermann_model_path": (str, None),
    # Optional extra input channels for SuperadiabaticNet (hybrid mode only) -
    # must match whatever the checkpoint being loaded was trained with, see
    # models/model.py:OPTIONAL_CHANNELS.
    "model_kwargs": (dict, {}),
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 0.1),
    "eta": (float, 0.67),
    "dt": (float, 1e-4),
    "rho_floor": (float, 1e-6),                 # Density clamp floor - must match the value the checkpoint was trained with for a fair comparison
    "t_range": (float, 4.0),                    # Total simulated time - lower for fast iteration (e.g. --t_range 0.5)
    "save_interval": (float, 0.01),
    "v_max_ceiling": (float, 50.0),              # Treat a finite-but-unphysical velocity as a blowup (physical v_max is typically 0.1-0.7)
    "max_wall_clock_seconds": (float, 1800.0),   # Hard stop so a CFL-shrunk-dt death spiral can't run forever (normal full run is ~17min)
}

if __name__ == "__main__":
    # 1. Load Config & Setup Paths
    args = load_config(PARAM_SCHEME, description="Modular Neural DDFT")
    
    current_time = datetime.now().strftime("%Y-%m-%d")
    base_path = f"results/{current_time}_{args['run_id']}_{args['model_mode']}"
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

    # 3. Load Models (frozen/eval - this is an evaluation script, nothing trains here)
    models, _ = build_models(
        mode=args['model_mode'], device=device, dx=dx,
        base_model_path=args["base_model_path"],
        sup_model_path=args["sup_model_path"],
        zimmermann_model_path=args["zimmermann_model_path"],
        trainable=False,
        sup_model_kwargs=args.get("model_kwargs", {}),
    )

    # 4. Initialize PhiFlow Grids & Solver
    rho = CenteredGrid(math.tensor(init_rho, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.tensor(V, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    solver = PhiFlowDDFT(
        models=models, device=device, dx=dx, config=args # Pass the whole config to access gamma, eta, k, T, m inside the step function
    )

    # 5. The Main Physics Loop
    t_range, save_interval, CFL, current_time, target_save = args['t_range'], args['save_interval'], 0.4, 0.0, 0.01
    step_counter = 0
    force_ratios = []
    blowup_info = {"occurred": False, "at_sim_time": None, "at_step": None, "reason": None}
    start_wall_time = time.time()
    rho_prev, v_prev = None, None  # no history available before the first step

    with tqdm(total=t_range, desc="DDFT Simulation", unit="sim_time", bar_format="{l_bar}{bar}| [{elapsed}<{remaining} {postfix}]") as pbar:
        while current_time < t_range:
            v_max = float(np.max(np.abs(v.values.vector['x'].native('batch,x').cpu().detach().numpy())))
            if np.isnan(v_max) or np.isinf(v_max):
                blowup_info = {"occurred": True, "at_sim_time": current_time, "at_step": step_counter, "reason": "v_max NaN/Inf"}
                break
            if v_max > args['v_max_ceiling']:
                # Finite but unphysical - without this, the CFL term below would keep
                # shrinking dt as v_max grows, making progress glacially slow without
                # ever hitting the NaN/Inf check (a "death spiral" that never finishes).
                blowup_info = {"occurred": True, "at_sim_time": current_time, "at_step": step_counter,
                                "reason": f"v_max={v_max:.3e} exceeded ceiling={args['v_max_ceiling']}"}
                break
            if time.time() - start_wall_time > args['max_wall_clock_seconds']:
                blowup_info = {"occurred": True, "at_sim_time": current_time, "at_step": step_counter,
                                "reason": f"wall-clock timeout after {args['max_wall_clock_seconds']}s (last v_max={v_max:.3e}, dt={dt if step_counter else None})"}
                break

            dt = min(CFL * dx / (v_max + 1e-6), args['dt'], t_range - current_time)

            rho_next, v_next = solver.step(rho, v, dt, V_ext, rho_prev=rho_prev, v_prev=v_prev)
            rho_prev, v_prev = rho, v
            rho, v = rho_next, v_next
            step_counter += 1

            current_time += dt
            pbar.update(dt)
            pbar.set_postfix(dt=f"{dt:.2e}", vmax=f"{v_max:.4f}")

            # 6. Save State
            if current_time >= target_save or current_time >= t_range - 1e-8:
                state_info = solver.save_state(rho, v, V_ext, current_time, results_path)
                if state_info.get("force_ratio") is not None:
                    force_ratios.append(state_info["force_ratio"])
                target_save += save_interval

    wall_clock_seconds = time.time() - start_wall_time

    print("Simulation complete! Generating movie...")
    plot_movie(results_path, args['md_path'], args['ref_md_path'])

    print("Writing run summary...")
    ref_path = args.get('ref_md_path') or args.get('md_path')
    reference_comparison = summarize_run(results_path, ref_path, is_ref_md=True) if ref_path else None
    write_run_summary(
        results_path=results_path, config=args,
        force_ratios=force_ratios, blowup_info=blowup_info,
        n_steps_completed=step_counter, final_sim_time=current_time,
        wall_clock_seconds=wall_clock_seconds, reference_comparison=reference_comparison,
    )
    print(f"Run summary saved to {os.path.join(results_path, 'summary.json')}")
