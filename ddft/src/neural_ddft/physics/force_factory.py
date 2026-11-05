import torch
import math
from phi.torch.flow import *
import torch.nn.functional as F

def get_equilibrium_force(rho_grid, base_model, dx, device):
    rho_pt = rho_grid.values.native(rho_grid.values.shape).float().to(device)

    if rho_pt.dim() == 1:
        rho_pt = rho_pt.unsqueeze(0)

    if not rho_pt.requires_grad:
        rho_pt = rho_pt.detach().requires_grad_(True)
    
    rho_input = rho_pt.unsqueeze(1)
    is_training = base_model.training

    with torch.enable_grad():
        F_pred = base_model(rho_input)
        grad_out = torch.autograd.grad(F_pred.sum(), rho_pt, create_graph=is_training, allow_unused=True)[0]

        if grad_out is None:
            dFdrho_tensor = torch.zeros_like(rho_input, device=device) / dx
        else:
            dFdrho_tensor = grad_out / dx

    if not is_training:
        dFdrho_tensor = dFdrho_tensor.detach()
    
    dFdrho_tensor = dFdrho_tensor.view(rho_pt.shape[0], -1)
    dFdrho_grid = CenteredGrid(math.tensor(dFdrho_tensor, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)
    f_eq = - field.spatial_gradient(dFdrho_grid)
    return f_eq, dFdrho_grid

def get_hybrid_force(rho_grid, v_grid, base_model, sup_model, dx, device):
    f_eq, dFdrho_grid = get_equilibrium_force(rho_grid, base_model, dx, device)

    rho_pt = rho_grid.values.native(rho_grid.values.shape).float().to(device)
    v_pt = v_grid.values.vector['x'].native(v_grid.values.vector['x'].shape).float().to(device)

    if rho_pt.dim() == 1: 
        rho_pt = rho_pt.unsqueeze(0)
    if v_pt.dim() == 1: 
        v_pt = v_pt.unsqueeze(0)
        
    # Calculate gradients using PyTorch central differences (Periodic Boundary Conditions)
    # torch.roll(..., -1) shifts left (i+1), torch.roll(..., 1) shifts right (i-1)
    grad_rho_pt = (torch.roll(rho_pt, shifts=-1, dims=-1) - torch.roll(rho_pt, shifts=1, dims=-1)) / (2 * dx)
    grad_v_pt = (torch.roll(v_pt, shifts=-1, dims=-1) - torch.roll(v_pt, shifts=1, dims=-1)) / (2 * dx)
    
    if sup_model.training:
        f_sup_val = sup_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt)
    else:
        with torch.no_grad():
            f_sup_val = sup_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt)

    f_sup = StaggeredGrid(math.tensor(f_sup_val, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)
    return f_eq + f_sup, dFdrho_grid

def get_zimmermann_force(rho_grid, v_grid, base_model, z_model, dx, device):
    """ Wrapped Zimmermann Superadiabatic Force """
    rho = rho_grid.values.native(rho_grid.values.shape).float().to(device)
    v_z = v_grid.values.vector['x'].native(v_grid.values.vector['x'].shape).float().to(device)

    if rho.dim() == 1: 
        rho = rho.unsqueeze(0)
    if v_z.dim() == 1: 
        v_z = v_z.unsqueeze(0)
    
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
