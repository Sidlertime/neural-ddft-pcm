import torch
import numpy as np
from phi.torch.flow import *
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT
from neural_ddft.physics.force_factory import get_hybrid_force

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

def test_superadiabatic_net_all_optional_channels_forward_and_backward():
    """
    Verifies the REAL SuperadiabaticNet (not a mock) with every optional
    input channel (V_ext, rho_prev, v_prev, accel, gamma, eta) enabled:
    forward pass produces the right shape, and a backward pass through
    get_hybrid_force actually reaches and updates every one of its weights -
    not just the last layer, the whole network, since autograd could in
    principle silently disconnect an earlier layer.
    """
    print("\n[DEBUG] Starting test_superadiabatic_net_all_optional_channels_forward_and_backward...")
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval().to(device)
    sup_model = SuperadiabaticNet(
        channels=8, use_vext=True, use_rho_prev=True, use_v_prev=True,
        use_accel=True, use_gamma=True, use_eta=True,
    ).train().to(device)
    # Force-field output weights are zero-initialized by design (stable start) -
    # perturb them so this test can observe a *changing* gradient/output,
    # not just "technically non-None".
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)

    rho = CenteredGrid(math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    rho_prev = CenteredGrid(0.9 * math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v_prev = StaggeredGrid(math.zeros(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    # V_ext is per-batch-sample in real usage too (different potentials in a
    # mixed batch have different external potentials) - match that here.
    V_ext = CenteredGrid(math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    f_total, _, components = get_hybrid_force(
        rho, v, base_model, sup_model, dx, device,
        V_ext_grid=V_ext, rho_prev_grid=rho_prev, v_prev_grid=v_prev, dt=1e-4,
        gamma=0.1, eta=0.6944,
    )
    f_sup_native = components["f_sup"].values.vector['x'].native(components["f_sup"].values.vector['x'].shape)
    assert f_sup_native.shape == (2, domain_res)

    loss = torch.sum(f_sup_native ** 2)
    loss.backward()

    for name, param in sup_model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter '{name}' has a non-finite gradient"
    print("[DEBUG] test_superadiabatic_net_all_optional_channels_forward_and_backward passed successfully.")


def test_superadiabatic_net_missing_required_channel_raises():
    print("\n[DEBUG] Starting test_superadiabatic_net_missing_required_channel_raises...")
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cpu")

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    sup_model = SuperadiabaticNet(channels=4, use_vext=True).train()

    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    # V_ext_grid is required (use_vext=True) but intentionally not passed.
    try:
        get_hybrid_force(rho, v, base_model, sup_model, dx, device)
        assert False, "Expected a ValueError for the missing V_ext"
    except ValueError as e:
        assert "V_ext" in str(e)
    print("[DEBUG] test_superadiabatic_net_missing_required_channel_raises passed successfully.")


def test_save_state_works_with_use_accel_and_no_history():
    """
    Regression test: PhiFlowDDFT.save_state() is called standalone (no
    rho_prev/v_prev tracked across calls, unlike the rollout loops in
    train_phiflow.py/phiflow_eval.py) - a sup_model with use_accel=True must
    not require a real dt there. This previously raised
    "sup_model requires accel but dt was not passed" because save_state
    didn't pass dt at all.
    """
    print("\n[DEBUG] Starting test_save_state_works_with_use_accel_and_no_history...")
    import tempfile, os
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cpu")

    config = {"model_mode": "hybrid", "m": 1.0, "gamma": 0.1, "eta": 0.67, "k": 1.0, "T": 2.0}

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    sup_model = SuperadiabaticNet(channels=4, use_vext=True, use_accel=True).eval()
    models = {"base": base_model, "sup": sup_model}

    rho = CenteredGrid(math.ones(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(math.zeros(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_info = solver.save_state(rho, v, V_ext, 0.0, tmpdir)
        assert state_info["force_ratio"] is not None
        assert os.path.exists(os.path.join(tmpdir, "f_sup00.000.npy"))
    print("[DEBUG] test_save_state_works_with_use_accel_and_no_history passed successfully.")


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
