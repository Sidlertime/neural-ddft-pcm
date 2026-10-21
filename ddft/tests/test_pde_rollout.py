import torch
import numpy as np
from phi.torch.flow import *
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT

def test_pde_step_execution():
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cpu")
    
    config = {
        "model_mode": "equilibrium",
        "m": 1.0,
        "gamma": 0.1,
        "eta": 0.67,
        "k": 1.0,
        "T": 2.0
    }
    
    # Simple mock base model
    class MockBaseModel(torch.nn.Module):
        def forward(self, rho):
            return torch.zeros_like(rho)
            
    models = {
        "base": MockBaseModel()
    }
    
    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)
    rho_next, v_next = solver.step(rho, v, 0.001, V_ext)
    
    assert rho_next is not None
    assert v_next is not None
