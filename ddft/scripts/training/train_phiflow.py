import os
import glob
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from phi.torch.flow import *

# Package imports
from neural_ddft.utils.config import load_config, get_next_result_path, save_config
from neural_ddft.data.dataset import MultiPotentialMDDataset, potential_name_from_path
from neural_ddft.models.factory import build_models, MODEL_KEY_BY_MODE
from neural_ddft.physics.force_factory import FORCE_MODES

# ==========================================
# Configuration Scheme
# ==========================================
PARAM_SCHEME = {
    "run_id": (str, "train"),
    "seed": (int, 42),                          # Controls chunk-start sampling and DataLoader shuffling
    "model_mode": (str, "hybrid"),              # "equilibrium", "hybrid", or "zimmermann"
    # One or more MD directories. Each becomes one "potential" in the mixed
    # training set; all must share the same spatial grid (asserted at load time).
    "md_paths": (list, ["../data/lj_data/ref_gaussian_langevin_01/avg_500/"]),
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
    "loss_weight_v": (float, 10.0),              # Balance between density loss and velocity loss
    "noise_std": (float, 1e-2),                  # Gaussian noise standard deviation for input perturbation (scheduled sampling / robustness)
    "curriculum": (list, [
        # one step prediction
        {"epochs": 10,  "window_size": 2,  "batch_size": 8, "lr": 1e-3, "stride": 1},
        {"epochs": 10,  "window_size": 2,  "batch_size": 8, "lr": 1e-3, "stride": 1},
        {"epochs": 10,  "window_size": 2,  "batch_size": 8, "lr": 1e-3, "stride": 1},
        {"epochs": 10,  "window_size": 2,  "batch_size": 8, "lr": 1e-3, "stride": 1},
        {"epochs": 10,  "window_size": 2,  "batch_size": 8, "lr": 1e-3, "stride": 1},
        # small window
        {"epochs": 10, "window_size": 5,  "batch_size": 8, "lr": 5e-4, "stride": 2},
        {"epochs": 10, "window_size": 5,  "batch_size": 8, "lr": 5e-4, "stride": 2},
        {"epochs": 10, "window_size": 5,  "batch_size": 8, "lr": 5e-4, "stride": 2},
        {"epochs": 10, "window_size": 5,  "batch_size": 8, "lr": 5e-4, "stride": 2},
        # medium horizon
        {"epochs": 10, "window_size": 15, "batch_size": 4, "lr": 2e-4, "stride": 6},
        {"epochs": 10, "window_size": 15, "batch_size": 4, "lr": 2e-4, "stride": 6},
        {"epochs": 10, "window_size": 15, "batch_size": 4, "lr": 2e-4, "stride": 6},
        # extended horizon
        {"epochs": 10, "window_size": 35, "batch_size": 1, "lr": 1e-4, "stride": 15},
        {"epochs": 10, "window_size": 35, "batch_size": 1, "lr": 1e-4, "stride": 15},
    ])
}

# ==========================================
# PhiFlow Differentiable PDE Step
# ==========================================
def pde_step(rho, v, dt, V_ext, compute_internal_force, models, dx, device, config, alpha=5.0):
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
    f_int, _, force_components = compute_internal_force(rho_next, v, models, dx, device)

    v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + (f_int * alpha / m))

    return rho_next, v_next, force_components

def scatter_add_by_potential(values_per_sample, potential_ids, accumulator):
    """
    Adds a (B,) per-sample tensor into a (n_potentials,) numpy accumulator,
    grouped by each sample's potential_id. Used to break down aggregate
    training metrics by potential when a batch mixes multiple potentials.
    """
    for pid in potential_ids.unique():
        mask = potential_ids == pid
        accumulator[int(pid)] += values_per_sample[mask].sum().item()
    return accumulator

# ==========================================
# Main Training Function
# ==========================================
def main():
    args = load_config(PARAM_SCHEME, description="Unified Neural DDFT Training Loop")
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Results & Checkpoint directory
    current_time = datetime.now().strftime("%Y-%m-%d")
    base_results = f"results/{current_time}_{args['run_id']}_{args['model_mode']}"
    run_dir = get_next_result_path(base_results)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(args, run_dir)

    # Potential names (one per md_paths entry) used to break down per-epoch
    # metrics per potential below - plain string manipulation, matches
    # MultiPotentialMDDataset's own naming logic, no dataset load needed.
    potential_names = [potential_name_from_path(p) for p in args["md_paths"]]

    # Initialize CSV logger with force ratio tracking (aggregate + per-potential columns)
    csv_log_path = os.path.join(run_dir, "training_log.csv")
    csv_headers = ["epoch", "phase", "window_size", "batch_size", "lr", "loss_total", "loss_rho", "loss_v", "force_ratio"]
    for name in potential_names:
        csv_headers += [f"loss_rho__{name}", f"loss_v__{name}", f"force_ratio__{name}", f"n_samples__{name}"]
    with open(csv_log_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)

    # Load Spatial Domain Dimensions from the first MD potential's files.
    # (All potentials must share this grid - MultiPotentialMDDataset asserts it below.)
    md_file_list = sorted(glob.glob(f"{args['md_paths'][0]}/collection.*.h5"))
    if not md_file_list:
        raise FileNotFoundError(f"No collection files found in {args['md_paths'][0]}")

    with h5py.File(md_file_list[0], 'r') as f:
        r_np = np.array(f['r']).flatten()

    box_length = float(r_np[-1] - r_np[0])
    domain_res = r_np.shape[0]
    dx = box_length / domain_res

    # Initialize Models Dictionary
    mode = args["model_mode"]
    models, trainable_params = build_models(
        mode=mode, device=device, dx=dx,
        base_model_path=args["base_model_path"],
        sup_model_path=args["sup_model_path"],
        zimmermann_model_path=args["zimmermann_model_path"],
        trainable=True,
    )

    compute_internal_force = FORCE_MODES[mode]
    criterion = nn.MSELoss()

    # If no parameters to optimize (e.g. pure equilibrium baseline)
    if not trainable_params:
        print(f"No trainable parameters found for mode '{mode}'. Exiting.")
        return

    base_weight_decay = float(args.get("weight_decay", 1e-4))
    base_loss_weight_v = float(args.get("loss_weight_v", 5.0))
    base_noise_std = float(args.get("noise_std", 0.0))

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=base_weight_decay)
    global_epoch = 0

    print(f"\nStarting training for mode: {mode}")
    print(f"gamma={args['gamma']} | eta={args['eta']} | base_weight_decay={base_weight_decay} | base_loss_weight_v={base_loss_weight_v}")
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
        
        # Allow per-phase parameter overrides
        phase_weight_decay = float(phase.get("weight_decay", base_weight_decay))
        phase_loss_weight_v = float(phase.get("loss_weight_v", base_loss_weight_v))
        phase_noise_std = float(phase.get("noise_std", base_noise_std))

        print(f"\n{'='*60}")
        print(f"--- Phase {phase_idx+1}/{len(args['curriculum'])}: Window Size={window_size}, Batch={batch_size}, LR={lr}, LossW_V={phase_loss_weight_v}, Noise={phase_noise_std} ---")
        print(f"{'='*60}")

        # Update learning rate and weight decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = phase_weight_decay

        full_dataset = MultiPotentialMDDataset(md_paths=args["md_paths"], window_size=window_size, stride=stride, sigma_smooth=3.0)

        # Pick one continuous 15% chunk PER POTENTIAL for this phase (never a
        # chunk spanning two potentials), then combine them into one subset.
        phase_indices = full_dataset.phase_chunk_indices(chunk_fraction=0.15)
        phase_subset = Subset(full_dataset, phase_indices)

        # Initialize DataLoader for this phase's specific chunk
        dataloader = DataLoader(phase_subset, batch_size=batch_size, shuffle=True, drop_last=True)

        for ep in range(epochs):
            global_epoch += 1
            epoch_loss = 0.0
            epoch_loss_rho = 0.0
            epoch_loss_v = 0.0
            epoch_force_ratio = 0.0
            num_batches = 0

            n_pot = len(potential_names)
            epoch_loss_rho_per_pot = np.zeros(n_pot)
            epoch_loss_v_per_pot = np.zeros(n_pot)
            epoch_force_ratio_per_pot = np.zeros(n_pot)
            epoch_n_samples_per_pot = np.zeros(n_pot)

            with tqdm(dataloader, desc=f"Epoch {global_epoch:03d} (Phase {phase_idx+1})", unit="batch") as pbar:
                for batch_data in pbar:
                    optimizer.zero_grad()

                    # batch['rho'] and batch['v'] shape: (B, window_size + 1, spatial_res)
                    rho_batch = batch_data['rho'].to(device)
                    v_batch = batch_data['v'].to(device)
                    potential_ids = batch_data['potential_id']
                    epoch_n_samples_per_pot += np.bincount(potential_ids.numpy(), minlength=n_pot)

                    # Each sample carries its own external potential V_ext, since
                    # a mixed batch can contain samples from different potentials.
                    V_ext_batch = full_dataset.V_ext[batch_data['potential_id']].to(device)
                    V_ext_grid = CenteredGrid(
                        math.tensor(V_ext_batch, batch('batch'), spatial('x')),
                        extrapolation.PERIODIC, bounds=Box(x=box_length)
                    )

                    # Optional noise injection for robustness against autoregressive exposure bias
                    if phase_noise_std > 0.0:
                        rho_batch = rho_batch + torch.randn_like(rho_batch) * phase_noise_std
                        rho_batch = torch.clamp(rho_batch, min=1e-6)

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
                    batch_force_ratio = torch.tensor(0.0, device=device)

                    # Unroll trajectory through the PDE
                    for step_idx in range(1, window_size + 1):
                        rho_next, v_next, force_components = pde_step(
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

                        loss = loss + loss_rho_step + (phase_loss_weight_v * loss_v_step)

                        # Per-sample (unreduced) losses purely for per-potential observability -
                        # does not affect the backprop target `loss` above.
                        with torch.no_grad():
                            loss_rho_per_sample = F.mse_loss(pred_rho, target_rho, reduction='none').mean(dim=-1)
                            loss_v_per_sample = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=-1)
                            scatter_add_by_potential(loss_rho_per_sample, potential_ids, epoch_loss_rho_per_pot)
                            scatter_add_by_potential(loss_v_per_sample, potential_ids, epoch_loss_v_per_pot)

                        # Compute relative force magnitude ratio (||f_sup|| / ||f_eq||) if hybrid mode.
                        # Reuses the f_eq/f_sup already computed inside pde_step's internal
                        # compute_internal_force call - no second base/sup model forward pass.
                        if mode == "hybrid":
                            f_eq_grid = force_components["f_eq"]
                            f_sup_grid = force_components["f_sup"]
                            f_eq_tensor = f_eq_grid.values.vector['x'].native(f_eq_grid.values.vector['x'].shape)
                            f_sup_tensor = f_sup_grid.values.vector['x'].native(f_sup_grid.values.vector['x'].shape)

                            norm_eq = torch.norm(f_eq_tensor) + 1e-8
                            norm_sup = torch.norm(f_sup_tensor)
                            ratio_step = (norm_sup / norm_eq).detach()
                            batch_force_ratio = batch_force_ratio + ratio_step

                            with torch.no_grad():
                                ratio_per_sample = torch.norm(f_sup_tensor, dim=-1) / (torch.norm(f_eq_tensor, dim=-1) + 1e-8)
                                scatter_add_by_potential(ratio_per_sample, potential_ids, epoch_force_ratio_per_pot)

                            # add regularization to network not overdominate force term (only tiny corrections)
                            force_penalty = torch.mean(f_sup_tensor ** 2)
                            loss = loss + (1e-4 * force_penalty)

                        rho_current = rho_next
                        v_current = v_next

                    loss.backward()

                    # Gradient clipping to prevent exploding gradients during long rollouts
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.detach().item()
                    epoch_loss_rho += batch_loss_rho.detach().item()
                    epoch_loss_v += batch_loss_v.detach().item()
                    epoch_force_ratio += (batch_force_ratio.detach().item() / window_size) if mode == "hybrid" else 0.0
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
            avg_ratio = epoch_force_ratio / max(num_batches, 1)

            # Per-potential denominators mirror n_denom (sample count * window_size)
            per_pot_denom = np.maximum(epoch_n_samples_per_pot * window_size, 1)
            avg_rho_per_pot = epoch_loss_rho_per_pot / per_pot_denom
            avg_v_per_pot = epoch_loss_v_per_pot / per_pot_denom
            avg_ratio_per_pot = epoch_force_ratio_per_pot / per_pot_denom

            ratio_str = " | ".join(f"{name}={avg_ratio_per_pot[i]:.4f}" for i, name in enumerate(potential_names)) if mode == "hybrid" else ""
            print(f"Epoch {global_epoch:03d} Summary | Mean Step: Total={avg_loss:.6f} | Rho={avg_rho:.6f} | V={avg_v:.6f} | ForceRatio={avg_ratio:.4f}" + (f" | PerPotentialRatio: {ratio_str}" if ratio_str else ""))

            # Append to CSV logger
            with open(csv_log_path, mode="a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                row = [global_epoch, phase_idx + 1, window_size, batch_size, lr, f"{avg_loss:.6e}", f"{avg_rho:.6e}", f"{avg_v:.6e}", f"{avg_ratio:.6e}"]
                for i in range(len(potential_names)):
                    row += [f"{avg_rho_per_pot[i]:.6e}", f"{avg_v_per_pot[i]:.6e}", f"{avg_ratio_per_pot[i]:.6e}", int(epoch_n_samples_per_pot[i])]
                writer.writerow(row)

            # Save checkpoint per epoch or on final phase step
            if global_epoch % 5 == 0 or ep == epochs - 1:
                ckpt_path = os.path.join(ckpt_dir, f"{mode}_epoch_{global_epoch:03d}.ckpt")
                active_model = models[MODEL_KEY_BY_MODE[mode]]
                torch.save(active_model.state_dict(), ckpt_path)

    print(f"\nTraining complete. All checkpoints saved in {ckpt_dir}")
    print(f"Training log written to {csv_log_path}")

if __name__ == "__main__":
    main()
