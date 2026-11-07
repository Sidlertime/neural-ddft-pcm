import builtins
import torch
import math
from phi.torch.flow import *
import torch.nn.functional as F
# from phi.torch.flow import * shadows Python's builtin round() with PhiFlow's
# own (single-argument) round_ - _smooth_periodic below needs the real one.

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

def _grid_to_native(grid, device, is_vector=False):
    values = grid.values.vector['x'] if is_vector else grid.values
    pt = values.native(values.shape).float().to(device)
    return pt.unsqueeze(0) if pt.dim() == 1 else pt


_GAUSSIAN_KERNEL_CACHE = {}

def _smooth_periodic(x_bt, sigma, device):
    """
    Applies a fixed (non-learned), differentiable Gaussian blur along the
    spatial dimension of a (B, X) tensor, with circular (periodic) boundary
    conditions matching this codebase's domain convention. Used to tame the
    use_potential_output path: a direct, deterministic countermeasure to the
    hypothesis that spatial_gradient amplifies whatever high-frequency
    content is present in the raw network output (see memory:
    scalar_potential_architecture.md) - smoothing it first removes that
    content before it gets amplified, at zero learned-parameter cost.
    """
    if sigma <= 0:
        return x_bt
    key = (builtins.round(sigma, 4), device)
    kernel = _GAUSSIAN_KERNEL_CACHE.get(key)
    if kernel is None:
        radius = max(1, int(builtins.round(3 * sigma)))
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        _GAUSSIAN_KERNEL_CACHE[key] = kernel
    pad = kernel.shape[-1] // 2
    x_padded = F.pad(x_bt.unsqueeze(1), (pad, pad), mode='circular')
    return F.conv1d(x_padded, kernel).squeeze(1)

def get_hybrid_force(rho_grid, v_grid, base_model, sup_model, dx, device,
                      V_ext_grid=None, rho_prev_grid=None, v_prev_grid=None, dt=None,
                      gamma=None, eta=None):
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

    # Build whichever optional input channels this particular sup_model was
    # configured to use (see models/model.py:OPTIONAL_CHANNELS) - absent
    # tensors fall back to a value that makes the corresponding channel
    # informationless (e.g. rho_prev defaults to rho_pt itself, giving zero
    # "acceleration" for the very first step of a window, where no real
    # history is available) rather than raising, except where no sane
    # fallback exists (V_ext, gamma, eta - the caller must always provide
    # these when the model needs them).
    extra_kwargs = {}
    if getattr(sup_model, "use_vext", False):
        if V_ext_grid is None:
            raise ValueError("sup_model requires V_ext but none was passed to get_hybrid_force")
        extra_kwargs["v_ext"] = _grid_to_native(V_ext_grid, device)

    needs_rho_prev = getattr(sup_model, "use_rho_prev", False)
    needs_v_prev = getattr(sup_model, "use_v_prev", False)
    needs_accel = getattr(sup_model, "use_accel", False)
    if needs_rho_prev or needs_accel:
        rho_prev_pt = _grid_to_native(rho_prev_grid, device) if rho_prev_grid is not None else rho_pt
        if needs_rho_prev:
            extra_kwargs["rho_prev"] = rho_prev_pt
    if needs_v_prev or needs_accel:
        v_prev_pt = _grid_to_native(v_prev_grid, device, is_vector=True) if v_prev_grid is not None else v_pt
        if needs_v_prev:
            extra_kwargs["v_prev"] = v_prev_pt
    if needs_accel:
        if dt is None or dt == 0:
            raise ValueError("sup_model requires accel but dt was not passed to get_hybrid_force")
        extra_kwargs["accel"] = (v_pt - v_prev_pt) / dt

    if getattr(sup_model, "use_gamma", False):
        if gamma is None:
            raise ValueError("sup_model requires gamma but none was passed to get_hybrid_force")
        extra_kwargs["gamma"] = torch.full_like(rho_pt, float(gamma))
    if getattr(sup_model, "use_eta", False):
        if eta is None:
            raise ValueError("sup_model requires eta but none was passed to get_hybrid_force")
        extra_kwargs["eta"] = torch.full_like(rho_pt, float(eta))

    if sup_model.training:
        sup_raw = sup_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt, **extra_kwargs)
    else:
        with torch.no_grad():
            sup_raw = sup_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt, **extra_kwargs)

    if getattr(sup_model, "use_potential_output", False):
        # sup_raw is a scalar superadiabatic potential field Phi_sup(z), not
        # yet a force - mirror get_equilibrium_force's second step exactly:
        # one ordinary (autograd-transparent) spatial_gradient, no
        # torch.autograd.grad needed here since sup_raw is already a
        # per-position field (SuperadiabaticNet has no pooling), unlike
        # F_eq's pooled-to-scalar CNN output.
        smooth_sigma = getattr(sup_model, "potential_smooth_sigma", 0.0)
        if smooth_sigma > 0:
            sup_raw = _smooth_periodic(sup_raw, smooth_sigma, device)
        sup_potential_grid = CenteredGrid(math.tensor(sup_raw, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)
        f_sup = - field.spatial_gradient(sup_potential_grid)
    else:
        f_sup = StaggeredGrid(math.tensor(sup_raw, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)

    return f_eq + f_sup, dFdrho_grid, {"f_eq": f_eq, "f_sup": f_sup}

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

    return f_total, dFdrho_grid, {"f_eq": f_eq, "f_zimmermann": f_zimmermann_grid}

def get_hybrid_power_force(rho_grid, v_grid, base_model, power_model, dx, device,
                            V_ext_grid=None, rho_prev_grid=None, v_prev_grid=None, dt=None,
                            gamma=None, eta=None):
    """
    f_sup = -(1/rho) * dP^ex/dv, where P^ex[rho, v] is a true scalar excess
    power functional predicted by power_model (PowerFunctionalNet) - the
    Power-Functional-Theory analog of F_eq, but differentiated w.r.t.
    velocity (not density) and needing no second spatial derivative.
    """
    f_eq, dFdrho_grid = get_equilibrium_force(rho_grid, base_model, dx, device)

    rho_pt = rho_grid.values.native(rho_grid.values.shape).float().to(device)
    v_pt = v_grid.values.vector['x'].native(v_grid.values.vector['x'].shape).float().to(device)

    if rho_pt.dim() == 1:
        rho_pt = rho_pt.unsqueeze(0)
    if v_pt.dim() == 1:
        v_pt = v_pt.unsqueeze(0)

    # v_pt must carry requires_grad BEFORE grad_v_pt is derived from it, so
    # torch.autograd.grad below sees both the direct-v and the grad-v input
    # paths (grad_v_pt is a torch.roll of v_pt, already differentiable).
    if not v_pt.requires_grad:
        v_pt = v_pt.detach().requires_grad_(True)

    grad_rho_pt = (torch.roll(rho_pt, shifts=-1, dims=-1) - torch.roll(rho_pt, shifts=1, dims=-1)) / (2 * dx)
    grad_v_pt = (torch.roll(v_pt, shifts=-1, dims=-1) - torch.roll(v_pt, shifts=1, dims=-1)) / (2 * dx)

    # Same optional-channel-building logic as get_hybrid_force (see that
    # function's comment for the fallback rationale).
    extra_kwargs = {}
    if getattr(power_model, "use_vext", False):
        if V_ext_grid is None:
            raise ValueError("power_model requires V_ext but none was passed to get_hybrid_power_force")
        extra_kwargs["v_ext"] = _grid_to_native(V_ext_grid, device)

    needs_rho_prev = getattr(power_model, "use_rho_prev", False)
    needs_v_prev = getattr(power_model, "use_v_prev", False)
    needs_accel = getattr(power_model, "use_accel", False)
    if needs_rho_prev or needs_accel:
        rho_prev_pt = _grid_to_native(rho_prev_grid, device) if rho_prev_grid is not None else rho_pt
        if needs_rho_prev:
            extra_kwargs["rho_prev"] = rho_prev_pt
    if needs_v_prev or needs_accel:
        v_prev_pt = _grid_to_native(v_prev_grid, device, is_vector=True) if v_prev_grid is not None else v_pt
        if needs_v_prev:
            extra_kwargs["v_prev"] = v_prev_pt
    if needs_accel:
        if dt is None or dt == 0:
            raise ValueError("power_model requires accel but dt was not passed to get_hybrid_power_force")
        extra_kwargs["accel"] = (v_pt - v_prev_pt) / dt

    if getattr(power_model, "use_gamma", False):
        if gamma is None:
            raise ValueError("power_model requires gamma but none was passed to get_hybrid_power_force")
        extra_kwargs["gamma"] = torch.full_like(rho_pt, float(gamma))
    if getattr(power_model, "use_eta", False):
        if eta is None:
            raise ValueError("power_model requires eta but none was passed to get_hybrid_power_force")
        extra_kwargs["eta"] = torch.full_like(rho_pt, float(eta))

    is_training = power_model.training
    with torch.enable_grad():
        p_exc = power_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt, **extra_kwargs)
        grad_out = torch.autograd.grad(p_exc.sum(), v_pt, create_graph=is_training, allow_unused=True)[0]
        dPdv_tensor = torch.zeros_like(v_pt) if grad_out is None else grad_out

    if not is_training:
        dPdv_tensor = dPdv_tensor.detach()

    # f_sup = -(1/rho) dP^ex/dv reintroduces the exact 1/rho singularity
    # mechanism root-caused for 3b/3d (weekend_ablation_sweep.md Round 6:
    # viscosity/pressure terms blow up as rho erodes toward its floor) -
    # clamp defensively from day one. This is a local safety clamp only
    # (not wired to the project's configurable rho_floor, which lives in
    # pde_step/PhiFlowDDFT and isn't threaded through FORCE_MODES' fixed
    # lambda signatures - adding it there would require every mode's
    # function to accept an extra kwarg it ignores, not worth the risk to
    # working modes for what is here just a divide-by-zero guard).
    _RHO_FLOOR_SAFETY_CLAMP = 1e-6
    f_sup_tensor = -dPdv_tensor / torch.clamp(rho_pt, min=_RHO_FLOOR_SAFETY_CLAMP)
    f_sup = StaggeredGrid(math.tensor(f_sup_tensor, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho_grid.bounds)

    return f_eq + f_sup, dFdrho_grid, {"f_eq": f_eq, "f_power": f_sup}

# The Dispatcher Dictionary
# Every mode returns (f_total, dFdrho_grid, force_components), where
# force_components is a dict of the individual force contributions (e.g.
# 'f_eq'/'f_sup') so callers needing them (e.g. force-ratio logging) can
# reuse values already computed here instead of recomputing a second forward pass.
# All three accept **extra so callers can always pass V_ext_grid/rho_prev_grid/
# v_prev_grid/dt/gamma/eta (cheap - only "hybrid" with the corresponding
# sup_model flags enabled actually uses them; the other modes just ignore them).
FORCE_MODES = {
    "equilibrium":  lambda rho, v, models, dx, dev, **extra: (*get_equilibrium_force(rho, models['base'], dx, dev), {}),
    "hybrid":       lambda rho, v, models, dx, dev, **extra: get_hybrid_force(rho, v, models['base'], models['sup'], dx, dev, **extra),
    "zimmermann":   lambda rho, v, models, dx, dev, **extra: get_zimmermann_force(rho, v, models['base'], models['zimmermann'], dx, dev),
    "hybrid_power": lambda rho, v, models, dx, dev, **extra: get_hybrid_power_force(rho, v, models['base'], models['power'], dx, dev, **extra),
}
