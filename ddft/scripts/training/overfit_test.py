import torch
import torch.nn as nn
from phi.torch.flow import *
from neural_ddft.data.dataset import MDWindowDataset
from neural_ddft.models.factory import build_models, MODEL_KEY_BY_MODE
from neural_ddft.physics.force_factory import FORCE_MODES
from train_phiflow import pde_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the dataset and pick ONE specific window in the middle of the simulation
    dataset = MDWindowDataset(md_path="../data/lj_data/ref_gaussian_langevin_01/avg_500", window_size=35, stride=1, sigma_smooth=3.0)
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
    models, trainable_params = build_models(mode=mode, device=device, dx=dataset.dr, trainable=True)
    criterion = nn.MSELoss()

    # If no parameters to optimize (e.g. pure equilibrium baseline)
    if not trainable_params:
        print(f"No trainable parameters found for mode '{mode}'. Exiting.")
        return

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    model = models[MODEL_KEY_BY_MODE[mode]]

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
            rho_next, v_next, _force_components = pde_step(rho_current, v_current, dt, V_ext_grid,
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