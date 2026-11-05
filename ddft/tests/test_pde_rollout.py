import torch
import numpy as np
from phi.torch.flow import *
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT

def test_pde_step_execution():
    print("\n[DEBUG] Starting test_pde_step_execution (multi-step rollout & gradient flow)...")
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = {
        "model_mode": "equilibrium",
        "m": 1.0,
        "gamma": 0.1,
        "eta": 0.67,
        "k": 1.0,
        "T": 2.0
    }
    
    # Simple mock base model with trainable parameter
    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            # Multiply rho directly so autograd tracks its dependency
            return rho * 0.0 * self.weight
            
    base_model = MockBaseModel()
    models = {
        "base": base_model
    }
    
    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)
    
    # Perform 3-step rollout and backward pass
    current_rho = rho
    current_v = v
    dt = 0.001
    
    loss = 0.0
    for step in range(3):
        current_rho, current_v = solver.step(current_rho, current_v, dt, V_ext)
        step_loss = torch.sum(current_rho.values.native(current_rho.values.shape) ** 2)
        loss = loss + step_loss
        print(f"[DEBUG] Rollout step {step+1}: loss contribution = {step_loss.item():.6f}")

    loss.backward()
    print(f"[DEBUG] Backward pass complete. Base model weight grad: {base_model.weight.grad}")
    
    assert current_rho is not None
    assert current_v is not None
    assert base_model.weight.grad is not None
    print("[DEBUG] test_pde_step_execution passed successfully.")

def test_hybrid_gradient_flow():
    print("\n[DEBUG] Starting test_hybrid_gradient_flow...")
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = {
        "model_mode": "hybrid",
        "m": 1.0, "gamma": 0.1, "eta": 0.67, "k": 1.0, "T": 2.0
    }
    
    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    class MockSupModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho, v, grad_rho, grad_v):
            return (rho + v + grad_rho + grad_v) * self.weight
            
    base_model = MockBaseModel().eval() # Base model frozen
    sup_model = MockSupModel().train()
    models = {"base": base_model, "sup": sup_model}
    
    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)
    
    rho_next, v_next = solver.step(rho, v, 0.001, V_ext)
    
    # Dummy loss to force backward pass
    v_native = v_next.values.vector['x'].native(v_next.values.vector['x'].shape)
    loss = torch.sum(v_native ** 2)
    loss.backward()
    
    assert sup_model.weight.grad is not None
    print("[DEBUG] test_hybrid_gradient_flow passed! Gradients reach the superadiabatic network.")

def test_equilibrium_gradient_flow():
    print("\n[DEBUG] Starting test_equilibrium_gradient_flow...")
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = {
        "model_mode": "equilibrium",
        "m": 1.0, "gamma": 0.1, "eta": 0.67, "k": 1.0, "T": 2.0
    }
    
    class TrainableBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        def forward(self, rho):
            # Ensure differentiable dependency on parameters and input rho
            return rho * self.weight

    base_model = TrainableBaseModel().train()
    models = {"base": base_model}
    
    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    
    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)
    
    rho_next, v_next = solver.step(rho, v, 0.001, V_ext)
    
    # Dummy loss to force backward pass
    v_native = v_next.values.vector['x'].native(v_next.values.vector['x'].shape)
    loss = torch.sum(v_native ** 2)
    loss.backward()
    
    assert base_model.weight.grad is not None
    print(f"[DEBUG] test_equilibrium_gradient_flow passed! Base model weight grad: {base_model.weight.grad.item()}")
