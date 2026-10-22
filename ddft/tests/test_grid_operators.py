import torch
import numpy as np
from phi.torch.flow import *

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
