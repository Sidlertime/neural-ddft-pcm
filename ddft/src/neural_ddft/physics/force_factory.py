import torch
import math
from phi.torch.flow import *
import torch.nn.functional as F

def get_equilibrium_force(rho_grid, base_model, dx, device):
    rho_pt = rho_grid.values.native('batch,x').unsqueeze(1).float().to(device).requires_grad_(True)
    is_training = base_model.training

    with torch.enable_grad():
        F_pred = base_model(rho_pt)
        dFdrho = torch.autograd.grad(F_pred.sum(), rho_pt, create_graph=is_training)[0] / dx

    if not is_training:
        dFdrho = dFdrho.detach()
    
    dFdrho_grid = CenteredGrid(math.tensor(dFdrho.squeeze(1), batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)
    f_eq = - field.spatial_gradient(dFdrho_grid)
    return f_eq, dFdrho_grid

def get_hybrid_force(rho_grid, v_grid, base_model, sup_model, dx, device):
    f_eq, dFdrho_grid = get_equilibrium_force(rho_grid, base_model, dx, device)

    rho_pt = rho_grid.values.native('batch,x').unsqueeze(1).float().to(device)
    v_pt = v_grid.values.vector['x'].native('batch,x').unsqueeze(1).float().to(device)
    
    if sup_model.training:
        f_sup_val = sup_model(rho_pt, v_pt).squeeze(1)
    else:
        with torch.no_grad():
            f_sup_val = sup_model(rho_pt, v_pt).squeeze(1)

    f_sup = StaggeredGrid(math.tensor(f_sup_pred, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)
    return f_eq + f_sup, dFdrho_grid

def get_zimmermann_force(rho_grid, v_grid, base_model, z_model, dx, device):
    """ Wrapped Zimmermann Superadiabatic Force """

    rho = rho_grid.values.native('batch,x').float().to(device)
    v_z = v_grid.values.vector['x'].native('batch,x').float().to(device)
    
    # Create the 4-channel input [rho, v_x, v_y, v_z]
    # The network expects all three velocity components, so we pad x and y with zeros.
    B, N = rho.shape
    zeros = torch.zeros_like(rho)
    x_full = torch.stack([rho, zeros, zeros, v_z], dim=1) # Shape: (B, 4, N)
    
    # Apply Periodic Padding
    # We need a centered window of 561 points. 
    # Therefore, we pad 280 points on the left and 280 on the right using circular boundaries.
    x_padded = F.pad(x_full, (280, 280), mode='circular')   # Shape (B, 4, N + 560)
    windows = x_padded.unfold(2, 561, 1)                    # Shape: (B, 4, N, 561)
    windows_flat = windows.permute(0, 2, 1, 3).reshape(B * N, 4 , 561)
    
    if z_model.training:
        f_zimmermann_flat = z_model(windows_flat)[:, 2]
    else:
        with torch.no_grad():
            f_zimmermann_flat = z_model(windows_flat)[:, 2] # Outputs (B * N, 3) representing f_x, f_y, f_z, we keep only z-direction
    
    f_zimmermann = f_zimmermann_flat.view(B, N)
    f_zimmermann_grid = StaggeredGrid(math.tensor(f_zimmermann, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)

    # OPTION A: Zimmermann predicts the ENTIRE internal force (Original Paper) with dummy dFdrho grid
    #f_total = f_zimmermann_grid
    #dFdrho_grid = CenteredGrid(math.zeros(batch(batch=B), spatial(x=N)), extrapolation.PERIODIC, bounds=rho_grid.bounds)

    # OPTION B: Zimmermann predicts ONLY the Superadiabatic Force (Hybrid Approach)
    f_eq, dFdrho_grid = get_equilibrium_force(rho_grid, base_model, dx, device)
    f_total = f_eq + f_zimmermann_grid

    return f_total, dFdrho_grid

# The Dispatcher Dictionary
FORCE_MODES = {
    "equilibrium": lambda rho, v, models, dx, dev: get_equilibrium_force(rho, models['base'], dx, dev),
    "hybrid":      lambda rho, v, models, dx, dev: get_hybrid_force(rho, v, models['base'], models['sup'], dx, dev),
    "zimmermann":  lambda rho, v, models, dx, dev: get_zimmermann_force(rho, v, models['base'], models['zimmermann'], dx, dev)
}