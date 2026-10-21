import os
import glob
import json
import csv
import torch
import torch.nn as nn
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from phi.torch.flow import *

# Package imports
from neural_ddft.utils.config import load_config, get_next_result_path, save_config
from neural_ddft.data.dataset import MDWindowDataset
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.models.zimmermann_net import ZimmermannNet
from neural_ddft.models.neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from neural_ddft.physics.force_factory import FORCE_MODES

# ==========================================
# Configuration Scheme
# ==========================================
PARAM_SCHEME = {
    "run_id": (str, "train"),
    "model_mode": (str, "hybrid"),              # "equilibrium", "hybrid", or "zimmermann"
    "md_path": (str, "../data/lj_data/2026-01-21_lj_langevin_01/avg_500/"),
    "base_model_path": (str, "models/c2_dz32.ckpt"),
    "sup_model_path": (str, None),
    "zimmermann_model_path": (str, None),
    
    # Physics parameters
    "k": (float, 1.0),
    "T": (float, 2.0),
    "m": (float, 1.0),
    "gamma": (float, 0.1),                      # Friction coefficient (1.0 / Tdamp in LJ units)
    "eta": (float, 0.67),                       # Shear viscosity (LJ units)
    "dt": (float, 1e-4),
    
    # Training parameters
    "weight_decay": (float, 1e-4),              # L2 regularization for smooth spatial derivatives
    "loss_weight_v": (float, 1.0),              # Balance between density loss and velocity loss
    "curriculum": (list, [
        {"epochs": 15, "window_size": 3, "batch_size": 8, "lr": 1e-3, "stride": 2},
        {"epochs": 20, "window_size": 10, "batch_size": 4,  "lr": 5e-4, "stride": 4},
        {"epochs": 25, "window_size": 25, "batch_size": 1,  "lr": 1e-4, "stride": 10},
    ])
}

# ==========================================
# PhiFlow Differentiable PDE Step
# ==========================================
def pde_step(rho, v, dt, V_ext, compute_internal_force, models, dx, device, config, alpha=1.0):
    """
    Performs one explicit integration step with computational graph intact.
    """
    m = config["m"]
    gamma = config["gamma"]
    eta = config["eta"]
    kBT = config["k"] * config["T"]

    # 1. Continuity: ∂ρ/∂t = -∇(ρv)
    rho_faces = rho.at(v)
    flux_centered = rho_faces * v
    grad_rho = field.spatial_gradient(rho)
    D_art = 0.5 * dx * math.abs(v)
    total_flux = flux_centered - D_art * grad_rho

    rho_next = rho - dt * field.divergence(total_flux)
    rho_next = rho_next.with_values(math.maximum(rho_next.values, 1e-6))

    # 2. Momentum equation
    v_advected = advect.semi_lagrangian(v, v, dt)
    rho_faces_next = rho_next.at(v)

    fric_term = -gamma * v
    visc_term = (eta / (m * rho_faces_next)) * field.laplace(v)
    pressure_term = - (kBT / (m * rho_faces_next)) * field.spatial_gradient(rho_next)
    vext_term = - (1.0 / m) * field.spatial_gradient(V_ext)

    # Dynamic internal force from selected model
    f_int, _ = compute_internal_force(rho_next, v, models, dx, device)

    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + (f_int * alpha / m))

    return rho_next, v_next

# ==========================================
# Main Training Function
# ==========================================
def main():
    args = load_config(PARAM_SCHEME, description="Unified Neural DDFT Training Loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Results & Checkpoint directory
    current_time = datetime.now().strftime("%Y-%m-%d")
    base_results = f"results/{current_time}_{args['run_id']}_{args['model_mode']}"
    run_dir = get_next_result_path(base_results)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(args, run_dir)

    # Initialize CSV logger
    csv_log_path = os.path.join(run_dir, "training_log.csv")
    csv_headers = ["epoch", "phase", "window_size", "batch_size", "lr", "loss_total", "loss_rho", "loss_v"]
    with open(csv_log_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)

    # Load Spatial Domain Dimensions from MD files
    md_file_list = sorted(glob.glob(f"{args['md_path']}/collection.*.h5"))
    if not md_file_list:
        raise FileNotFoundError(f"No collection files found in {args['md_path']}")
        
    with h5py.File(md_file_list[0], 'r') as f:
        V_np = np.array(f['V']).flatten()
        r_np = np.array(f['r']).flatten()

    box_length = float(r_np[-1] - r_np[0])
    domain_res = r_np.shape[0]
    dx = box_length / domain_res
    V_ext_grid = CenteredGrid(math.tensor(V_np, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    # Initialize Models Dictionary
    models = {}
    trainable_params = []
    mode = args["model_mode"]

    if int(1 / dx) == 100:
        base_model = NeuralFreeEnergyFine.load_from_checkpoint(args["base_model_path"], map_location=device)
    else:
        base_model = NeuralFreeEnergy.load_from_checkpoint(args["base_model_path"], map_location=device)
    base_model.to(device)
    models["base"] = base_model

    if mode == "equilibrium":
        base_model.train()
        for p in base_model.parameters(): 
            p.requires_grad = True
        trainable_params.extend(base_model.parameters())
    else:
        base_model.eval()
        for p in base_model.parameters(): 
            p.requires_grad = False

    if mode == "hybrid":
        sup_model = SuperadiabaticNet(channels=16).to(device)
        if args["sup_model_path"]:
            sup_model.load_state_dict(torch.load(args["sup_model_path"], map_location=device))
        sup_model.train()
        trainable_params.extend(sup_model.parameters())
        models["sup"] = sup_model
    elif mode == "zimmermann":
        z_model = ZimmermannNet().to(device)
        if args["zimmermann_model_path"]:
            z_model.load_state_dict(torch.load(args["zimmermann_model_path"], map_location=device))
        z_model.train()
        trainable_params.extend(z_model.parameters())
        models["zimmermann"] = z_model

    compute_internal_force = FORCE_MODES[mode]
    criterion = nn.MSELoss()

    # If no parameters to optimize (e.g. pure equilibrium baseline)
    if not trainable_params:
        print(f"No trainable parameters found for mode '{mode}'. Exiting.")
        return

    weight_decay = float(args.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=weight_decay)
    global_epoch = 0

    print(f"\nStarting training for mode: {mode}")
    print(f"gamma={args['gamma']} | eta={args['eta']} | weight_decay={weight_decay} | loss_weight_v={args['loss_weight_v']}")
    print(f"Log CSV saved to: {csv_log_path}")
    print(f"Checkpoints will be saved to: {ckpt_dir}\n")

    # ==========================================
    # Curriculum Loop
    # ==========================================
    for phase_idx, phase in enumerate(args["curriculum"]):
        window_size = int(phase["window_size"])
        epochs = int(phase["epochs"])
        batch_size = int(phase["batch_size"])
        lr = float(phase["lr"])
        stride = int(phase.get("stride", 1))

        print(f"\n{'='*60}")
        print(f"--- Phase {phase_idx+1}/{len(args['curriculum'])}: Window Size={window_size}, Batch={batch_size}, LR={lr} ---")
        print(f"{'='*60}")

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        dataset = MDWindowDataset(md_path=args["md_path"], window_size=window_size, stride=stride, sigma_smooth=3.0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for ep in range(epochs):
            global_epoch += 1
            epoch_loss = 0.0
            epoch_loss_rho = 0.0
            epoch_loss_v = 0.0
            num_batches = 0

            with tqdm(dataloader, desc=f"Epoch {global_epoch:03d} (Phase {phase_idx+1})", unit="batch") as pbar:
                for batch_data in pbar:
                    optimizer.zero_grad()

                    # batch['rho'] and batch['v'] shape: (B, window_size + 1, spatial_res)
                    rho_batch = batch_data['rho'].to(device)
                    v_batch = batch_data['v'].to(device)
                    B = rho_batch.shape[0]

                    # Initial conditions (step 0 of window)
                    rho_current = CenteredGrid(
                        math.tensor(rho_batch[:, 0, :], batch('batch'), spatial('x')),
                        extrapolation.PERIODIC, bounds=Box(x=box_length)
                    )
                    v_current = StaggeredGrid(
                        math.tensor(v_batch[:, 0, :], batch('batch'), spatial('x')),
                        extrapolation.PERIODIC, bounds=Box(x=box_length)
                    )

                    loss = torch.tensor(0.0, device=device)
                    batch_loss_rho = torch.tensor(0.0, device=device)
                    batch_loss_v = torch.tensor(0.0, device=device)

                    # Unroll trajectory through the PDE
                    for step_idx in range(1, window_size + 1):
                        rho_next, v_next = pde_step(
                            rho_current, v_current, args["dt"], V_ext_grid,
                            compute_internal_force, models, dx, device, args
                        )

                        pred_rho = rho_next.values.native('batch,x')
                        pred_v = v_next.values.vector['x'].native('batch,x')

                        target_rho = rho_batch[:, step_idx, :]
                        target_v = v_batch[:, step_idx, :]

                        loss_rho_step = criterion(pred_rho, target_rho)
                        loss_v_step = criterion(pred_v, target_v)

                        batch_loss_rho = batch_loss_rho + loss_rho_step
                        batch_loss_v = batch_loss_v + loss_v_step

                        loss = loss + loss_rho_step + (args["loss_weight_v"] * loss_v_step)

                        rho_current = rho_next
                        v_current = v_next

                    loss.backward()

                    # Gradient clipping to prevent exploding gradients during long rollouts
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_loss_rho += batch_loss_rho.item()
                    epoch_loss_v += batch_loss_v.item()
                    num_batches += 1

                    pbar.set_postfix(
                        loss=f"{loss.item() / window_size:.5f}",
                        l_rho=f"{batch_loss_rho.item() / window_size:.5f}",
                        l_v=f"{batch_loss_v.item() / window_size:.5f}",
                    )

            n_denom = max(num_batches * window_size, 1)
            avg_loss = epoch_loss / n_denom
            avg_rho = epoch_loss_rho / n_denom
            avg_v = epoch_loss_v / n_denom
            print(f"Epoch {global_epoch:03d} Summary | Mean Step: Total={avg_loss:.6f} | Rho={avg_rho:.6f} | V={avg_v:.6f}")

            # Append to CSV logger
            with open(csv_log_path, mode="a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([global_epoch, phase_idx + 1, window_size, batch_size, lr, f"{avg_loss:.6e}", f"{avg_rho:.6e}", f"{avg_v:.6e}"])

            # Save checkpoint per epoch or on final phase step
            if global_epoch % 5 == 0 or ep == epochs - 1:
                ckpt_path = os.path.join(ckpt_dir, f"{mode}_epoch_{global_epoch:03d}.ckpt")
                active_model = models[{"equilibrium":"base","hybrid":"sup","zimmermann":"zimmermann"}[mode]]
                torch.save(active_model.state_dict(), ckpt_path)

    print(f"\nTraining complete. All checkpoints saved in {ckpt_dir}")
    print(f"Training log written to {csv_log_path}")

if __name__ == "__main__":
    main()
