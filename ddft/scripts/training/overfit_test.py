import torch
import torch.nn as nn
from phi.torch.flow import *
from neural_ddft.data.dataset import MDWindowDataset
from neural_ddft.models.neural_free_energy import NeuralFreeEnergy, NeuralFreeEnergyFine
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.models.zimmermann_net import ZimmermannNet
from neural_ddft.physics.force_factory import FORCE_MODES
from train_phiflow import pde_step

def model_setup(mode, device, dx, base_model_path="models/c2_dz32.ckpt", sup_model_path=None, zimmermann_model_path=None):
    models = {}
    trainable_params = []

    # Load base model robustly supporting raw state dict checkpoints as well as lightning checkpoints
    if int(1 / dx) == 100:
        base_model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path="models/c2_dz32.ckpt", map_location=device)
    else:
        base_model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path="models/c2_dz32.ckpt", map_location=device)

    try:
        ckpt_data = torch.load(base_model_path, map_location=device)
        if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            base_model.load_state_dict(ckpt_data["state_dict"])
        elif isinstance(ckpt_data, dict):
            base_model.load_state_dict(ckpt_data)
        print(f"Successfully loaded base model weights from {base_model_path}")
    except Exception as e:
        if base_model_path != "models/c2_dz32.ckpt":
            print(f"Warning: Could not load state dict from {base_model_path} ({e}), falling back to direct load_from_checkpoint.")
            if int(1/dx) == 100:
                base_model = NeuralFreeEnergyFine.load_from_checkpoint(checkpoint_path=base_model_path, map_location=device)
            else:
                base_model = NeuralFreeEnergy.load_from_checkpoint(checkpoint_path=base_model_path, map_location=device)
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
        if sup_model_path:
            sup_ckpt = torch.load(sup_model_path, map_location=device)
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
        if zimmermann_model_path:
            z_ckpt = torch.load(zimmermann_model_path, map_location=device)
            if isinstance(z_ckpt, dict) and "state_dict" in z_ckpt:
                z_model.load_state_dict(z_ckpt["state_dict"])
            elif isinstance(z_ckpt, dict):
                z_model.load_state_dict(z_ckpt)
            else:
                z_model.load_state_dict(z_ckpt)
        z_model.train()
        trainable_params.extend(z_model.parameters())
        models["zimmermann"] = z_model
    
    return models, trainable_params

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the dataset and pick ONE specific window in the middle of the simulation
    dataset = MDWindowDataset(md_path="../data/lj_data/2026-01-21_lj_langevin_01/avg_500", window_size=35, stride=1, sigma_smooth=3.0)
    target_idx = len(dataset) // 2  # Grab a snippet from the middle where velocity is active
    single_batch = dataset[target_idx]

    # Add batch dimension: (1, window_size+1, spatial_res)
    rho_batch = single_batch['rho'].unsqueeze(0).to(device)
    v_batch = single_batch['v'].unsqueeze(0).to(device)

    # params
    dt = 1e-4
    config = {
        "m": 1.0,
        "T": 2.0,
        "k": 1.0,
        "gamma": 0.1,
        "eta": 0.6944,
    }

    mode = "zimmermann"
    compute_internal_force = FORCE_MODES[mode]
    models, trainable_params = model_setup(mode, device, dataset.dr)
    criterion = nn.MSELoss()

    # If no parameters to optimize (e.g. pure equilibrium baseline)
    if not trainable_params:
        print(f"No trainable parameters found for mode '{mode}'. Exiting.")
        return

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    model_map = {'equilibrium': 'base', 'hybrid': 'sup', 'zimmermann': 'zimmermann'}
    model = models[model_map[mode]]

    V_ext_grid = CenteredGrid(math.tensor(dataset.V_ext.to(device), spatial('x')), extrapolation.PERIODIC, x=dataset.domain_resolution, bounds=Box(x=dataset.box_length))

    # 2. Hammer this single snippet for 500 epochs
    print(f"Training {mode} model:\n")
    for epoch in range(500):
        optimizer.zero_grad()
        
        rho_current = CenteredGrid(math.tensor(rho_batch[:, 0, :], batch('batch'), spatial('x')),
                            extrapolation.PERIODIC, bounds=Box(x=dataset.box_length)) # Init from rho_batch[:, 0, :]
        v_current = StaggeredGrid(math.tensor(v_batch[:, 0, :], batch('batch'), spatial('x')),
                            extrapolation.PERIODIC, bounds=Box(x=dataset.box_length))  # Init from v_batch[:, 0, :]
        
        loss = 0.0
        for step in range(1, 36):
            rho_next, v_next = pde_step(rho_current, v_current, dt, V_ext_grid,
                                compute_internal_force, models, dataset.dr, device, config)

            pred_rho = rho_next.values.native('batch,x')
            pred_v = v_next.values.vector['x'].native('batch,x')

            target_rho = rho_batch[:, step, :]
            target_v = v_batch[:, step, :]
            
            # Calculate loss against rho_batch[:, step, :]
            loss += criterion(pred_rho, target_rho) + 5.0 * criterion(pred_v, target_v)
            
            rho_current, v_current = rho_next, v_next
            
        loss.backward()
        
        # DEBUG: Check if gradients are actually flowing!
        if mode == "equilibrium":
            final_layer = model.model.output_net 
            if isinstance(final_layer, nn.Sequential):
                final_layer = final_layer[0]
            grad_norm = final_layer.weight.grad.norm().item()
        elif mode == "hybrid":
            grad_norm = model.net[-1].weight.grad.norm().item()
        elif mode == "zimmermann":
            grad_norm = model.fc2.weight.grad.norm().item()
        
        optimizer.step()
        
        print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.2e}")

if __name__=="__main__":
    main()