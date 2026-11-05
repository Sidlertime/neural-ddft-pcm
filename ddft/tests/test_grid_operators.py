import torch
import numpy as np
from phi.torch.flow import *
import math as std_math

def test_mass_conservation():
    print("\n[DEBUG] Starting test_mass_conservation...")
    box_length = 10.0
    domain_res = 64
    dx = box_length / domain_res
    
    init_rho = np.ones(domain_res) + 0.1 * np.sin(2 * np.pi * np.arange(domain_res) / domain_res)
    rho = CenteredGrid(math.tensor(init_rho, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.ones(spatial(x=domain_res)) * 0.1, extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    dt = 0.01
    initial_mass = float(math.sum(rho.values))
    print(f"[DEBUG] Initial total mass: {initial_mass:.6f}")

    # Multi-step evolution test
    current_rho = rho
    for step in range(5):
        rho_faces = current_rho.at(v)
        flux = rho_faces * v
        current_rho = current_rho - dt * field.divergence(flux)
        mass_step = float(math.sum(current_rho.values))
        print(f"[DEBUG] Step {step+1}: mass = {mass_step:.6f}")

    final_mass = float(math.sum(current_rho.values))
    print(f"[DEBUG] Final total mass: {final_mass:.6f}, Difference: {abs(initial_mass - final_mass):.2e}")
    
    assert np.isclose(initial_mass, final_mass, rtol=1e-5, atol=1e-5)
    print("[DEBUG] test_mass_conservation passed successfully.")

def test_manual_spatial_gradient():
    print("\n[DEBUG] Starting test_manual_spatial_gradient...")
    
    # 1. Setup a periodic domain from 0 to 2*pi
    domain_res = 128
    box_length = 2.0 * std_math.pi
    dx = box_length / domain_res
    
    # Create spatial coordinates (omitting the very last point to maintain perfect periodicity)
    x = torch.linspace(0, box_length - dx, domain_res)
    
    # 2. Create analytical functions: f(x) = sin(x), f'(x) = cos(x)
    rho_pt = torch.sin(x).unsqueeze(0)  # Shape (1, X)
    exact_grad = torch.cos(x).unsqueeze(0)
    
    # 3. Apply our manual central difference from force_factory.py
    grad_rho_pt = (torch.roll(rho_pt, shifts=-1, dims=-1) - torch.roll(rho_pt, shifts=1, dims=-1)) / (2 * dx)
    
    # 4. Calculate error
    mse_error = torch.mean((grad_rho_pt - exact_grad) ** 2).item()
    print(f"[DEBUG] Finite Difference vs Analytical MSE: {mse_error:.6e}")
    
    # 5. Assert that the discrete approximation is highly accurate
    # (Error scales with dx^2 for central differences, so it should be very small)
    assert mse_error < 1e-3, f"Gradient approximation error too high: {mse_error}"
    
    # Check periodic boundaries specifically (first and last elements)
    assert torch.isclose(grad_rho_pt[0, 0], exact_grad[0, 0], atol=1e-2), "Left boundary failed!"
    assert torch.isclose(grad_rho_pt[0, -1], exact_grad[0, -1], atol=1e-2), "Right boundary failed!"
    
    print("[DEBUG] test_manual_spatial_gradient passed successfully.")
