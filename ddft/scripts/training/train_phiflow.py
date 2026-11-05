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
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from phi.torch.flow import *

# Package imports
from neural_ddft.utils.config import load_config, get_next_result_path, save_config
from neural_ddft.data.dataset import MDWindowDataset
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.models.zimmermann_net import ZimmermannNet
from neural_ddft.models.neural_free_energy import NeuralFreeEnergyFine, NeuralFreeEnergy
from neural_ddft.physics.force_factory import FORCE_MODES, get_equilibrium_force

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

    # Initialize CSV logger with force ratio tracking
    csv_log_path = os.path.join(run_dir, "training_log.csv")
    csv_headers = ["epoch", "phase", "window_size", "batch_size", "lr", "loss_total", "loss_rho", "loss_v", "force_ratio"]
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

    # Load base model robustly supporting raw state dict checkpoints as well as lightning checkpoints
    if int(1 / dx) == 100:
        base_model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path="models/c2_dz32.ckpt", map_location=device)
    else:
        base_model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path="models/c2_dz32.ckpt", map_location=device)

    try:
        ckpt_data = torch.load(args["base_model_path"], map_location=device)
        if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            base_model.load_state_dict(ckpt_data["state_dict"])
        elif isinstance(ckpt_data, dict):
            base_model.load_state_dict(ckpt_data)
        print(f"Successfully loaded base model weights from {args['base_model_path']}")
    except Exception as e:
        if args["base_model_path"] != "models/c2_dz32.ckpt":
            print(f"Warning: Could not load state dict from {args['base_model_path']} ({e}), falling back to direct load_from_checkpoint.")
            if int(1/dx) == 100:
                base_model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=args["base_model_path"], map_location=device)
            else:
                base_model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=args["base_model_path"], map_location=device)
    base_model.to(device)

    if mode == "equilibrium":
        base_model.train()
        # Freeze all parameters first
        for p in base_model.parameters(): 
            p.requires_grad = False

        # Unfreeze ONLY the final layer
        # If NeuralFreeEnergy uses a Sequential block named 'net' ending in Conv1d/Linear
        final_layer = list(base_model.children())[-1] 
        if isinstance(final_layer, nn.Sequential):
            final_layer = final_layer[-1]
            
        for p in final_layer.parameters():
            p.requires_grad = True
            trainable_params.append(p)
            
        print("Equilibrium mode: Froze feature extractors. Fine-tuning ONLY the final layer.")
    else:
        base_model.eval()
        for p in base_model.parameters(): 
            p.requires_grad = False
    models["base"] = base_model

    if mode == "hybrid":
        sup_model = SuperadiabaticNet(channels=16).to(device)
        if args["sup_model_path"]:
            sup_ckpt = torch.load(args["sup_model_path"], map_location=device)
            if isinstance(sup_ckpt, dict) and "state_dict" in sup_ckpt:
                sup_model.load_state_dict(sup_ckpt["state_dict"])
            elif isinstance(sup_ckpt, dict):
                sup_model.load_state_dict(sup_ckpt)
            else:
                sup_model.load_state_dict(sup_ckpt)
        sup_model.train()
        trainable_params.extend(sup_model.parameters())
        models["sup"] = sup_model
    elif mode == "zimmermann":
        z_model = ZimmermannNet().to(device)
        if args["zimmermann_model_path"]:
            z_ckpt = torch.load(args["zimmermann_model_path"], map_location=device)
            if isinstance(z_ckpt, dict) and "state_dict" in z_ckpt:
                z_model.load_state_dict(z_ckpt["state_dict"])
            elif isinstance(z_ckpt, dict):
                z_model.load_state_dict(z_ckpt)
            else:
                z_model.load_state_dict(z_ckpt)
        z_model.train()
        trainable_params.extend(z_model.parameters())
        models["zimmermann"] = z_model

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

        full_dataset = MDWindowDataset(md_path=args["md_path"], window_size=window_size, stride=stride, sigma_smooth=3.0)
        
        # Define the continuous chunk size (e.g., 15% of the trajectory)
        chunk_size = max(1, int(0.15 * len(full_dataset)))
        max_start_idx = max(0, len(full_dataset) - chunk_size)

        # Randomly select the starting point for this PHASE
        start_idx = np.random.randint(0, max_start_idx + 1)
        phase_indices = list(range(start_idx, start_idx + chunk_size))
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

            with tqdm(dataloader, desc=f"Epoch {global_epoch:03d} (Phase {phase_idx+1})", unit="batch") as pbar:
                for batch_data in pbar:
                    optimizer.zero_grad()

                    # batch['rho'] and batch['v'] shape: (B, window_size + 1, spatial_res)
                    rho_batch = batch_data['rho'].to(device)
                    v_batch = batch_data['v'].to(device)
                    
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

                        loss = loss + loss_rho_step + (phase_loss_weight_v * loss_v_step)

                        # Compute relative force magnitude ratio (||f_sup|| / ||f_eq||) if hybrid mode
                        if mode == "hybrid":
                            try:
                                f_eq_grid, _ = get_equilibrium_force(rho_next, models["base"], dx, device)
                                f_eq_tensor = f_eq_grid.values.vector['x'].native(f_eq_grid.values.vector['x'].shape)

                                rho_pt = rho_next.values.native(rho_next.values.shape).float()
                                v_pt = v_next.values.vector['x'].native(v_next.values.vector['x'].shape).float()
                                grad_rho_pt = (torch.roll(rho_pt, shifts=-1, dims=-1) - torch.roll(rho_pt, shifts=1, dims=-1)) / (2 * dx)
                                grad_v_pt = (torch.roll(v_pt, shifts=-1, dims=-1) - torch.roll(v_pt, shifts=1, dims=-1)) / (2 * dx)
                                f_sup_tensor = models["sup"](rho_pt, v_pt, grad_rho_pt, grad_v_pt)

                                norm_eq = torch.norm(f_eq_tensor) + 1e-8
                                norm_sup = torch.norm(f_sup_tensor)
                                ratio_step = (norm_sup / norm_eq).detach()
                                batch_force_ratio = batch_force_ratio + ratio_step

                                # add regularization to network not overdominate force term (only tiny corrections)
                                force_penalty = torch.mean(f_sup_tensor ** 2)
                                loss = loss + (1e-4 * force_penalty)
                            except Exception as e:
                                print(f"Ratio calculation failed: {e}")

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
            print(f"Epoch {global_epoch:03d} Summary | Mean Step: Total={avg_loss:.6f} | Rho={avg_rho:.6f} | V={avg_v:.6f} | ForceRatio={avg_ratio:.4f}")

            # Append to CSV logger
            with open(csv_log_path, mode="a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([global_epoch, phase_idx + 1, window_size, batch_size, lr, f"{avg_loss:.6e}", f"{avg_rho:.6e}", f"{avg_v:.6e}", f"{avg_ratio:.6e}"])

            # Save checkpoint per epoch or on final phase step
            if global_epoch % 5 == 0 or ep == epochs - 1:
                ckpt_path = os.path.join(ckpt_dir, f"{mode}_epoch_{global_epoch:03d}.ckpt")
                active_model = models[{"equilibrium":"base","hybrid":"sup","zimmermann":"zimmermann"}[mode]]
                torch.save(active_model.state_dict(), ckpt_path)

    print(f"\nTraining complete. All checkpoints saved in {ckpt_dir}")
    print(f"Training log written to {csv_log_path}")

if __name__ == "__main__":
    main()
