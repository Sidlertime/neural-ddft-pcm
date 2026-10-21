import torch
import numpy as np
from phi.torch.flow import *

def test_mass_conservation():
    box_length = 10.0
    domain_res = 64
    dx = box_length / domain_res
    
    init_rho = np.ones(domain_res) + 0.1 * np.sin(2 * np.pi * np.arange(domain_res) / domain_res)
    rho = CenteredGrid(math.tensor(init_rho, spatial('x')), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.ones(spatial(x=domain_res)) * 0.1, extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    dt = 0.01
    rho_faces = rho.at(v)
    flux = rho_faces * v
    rho_next = rho - dt * field.divergence(flux)
    
    initial_mass = float(math.sum(rho.values))
    final_mass = float(math.sum(rho_next.values))
    
    assert np.isclose(initial_mass, final_mass, rtol=1e-5, atol=1e-5)
