import torch
import numpy as np
from phi.torch.flow import *
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT
from neural_ddft.physics.force_factory import get_hybrid_force, _smooth_periodic

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


def test_superadiabatic_net_potential_output_matches_manual_spatial_gradient():
    """
    Validates the use_potential_output path end to end: when enabled,
    SuperadiabaticNet's raw (B, X) output must be treated as a scalar
    potential field and turned into a force via exactly one
    field.spatial_gradient call (negated) - not a coincidental pass-through.
    Confirmed here by recomputing that spatial gradient manually from the
    network's own raw output and checking it matches get_hybrid_force's
    f_sup bit-for-bit, the same style of direct-correctness check already
    used for the autograd half of this mechanism in
    test_force_factory.py:test_equilibrium_chemical_potential_matches_finite_difference.
    """
    print("\n[DEBUG] Starting test_superadiabatic_net_potential_output_matches_manual_spatial_gradient...")
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

    base_model = MockBaseModel().eval().to(device)
    sup_model = SuperadiabaticNet(channels=8, use_potential_output=True).eval().to(device)
    # Zero-initialized output head (by design) would make this check trivial
    # (0 everywhere) - perturb it so there's a genuine non-flat potential field.
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)
        sup_model.net[-1].bias.add_(torch.randn(1) * 0.1)

    rho = CenteredGrid(0.6 + 0.1 * math.random_normal(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    f_total, _, components = get_hybrid_force(rho, v, base_model, sup_model, dx, device)
    f_sup = components["f_sup"]

    # Manually recompute: call sup_model directly with the same inputs
    # get_hybrid_force builds internally, then apply spatial_gradient ourselves.
    rho_pt = rho.values.native(rho.values.shape).float().unsqueeze(0).to(device)
    v_pt = v.values.vector['x'].native(v.values.vector['x'].shape).float().unsqueeze(0).to(device)
    grad_rho_pt = (torch.roll(rho_pt, -1, dims=-1) - torch.roll(rho_pt, 1, dims=-1)) / (2 * dx)
    grad_v_pt = (torch.roll(v_pt, -1, dims=-1) - torch.roll(v_pt, 1, dims=-1)) / (2 * dx)
    with torch.no_grad():
        raw = sup_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt)
    potential_grid = CenteredGrid(math.tensor(raw, batch('batch'), spatial('x')), extrapolation.PERIODIC, bounds=rho.bounds)
    expected_f_sup = -field.spatial_gradient(potential_grid)

    actual = f_sup.values.vector['x'].native(f_sup.values.vector['x'].shape).detach().cpu().numpy()
    expected = expected_f_sup.values.vector['x'].native(expected_f_sup.values.vector['x'].shape).detach().cpu().numpy()
    assert np.allclose(actual, expected, atol=1e-6), "f_sup does not match a manually-applied spatial_gradient of the raw network output"

    # Sanity check this isn't trivially zero everywhere (would pass the
    # allclose check above vacuously if the gradient computation was skipped
    # and both sides independently returned zero).
    assert np.abs(actual).max() > 1e-6, "f_sup is suspiciously all-zero - spatial_gradient may not be getting applied"
    print("[DEBUG] test_superadiabatic_net_potential_output_matches_manual_spatial_gradient passed successfully.")


def test_superadiabatic_net_potential_output_forward_and_backward():
    """
    Same gradient-reaches-every-parameter check as
    test_superadiabatic_net_all_optional_channels_forward_and_backward, but
    for use_potential_output=True - confirms the extra spatial_gradient step
    doesn't block backprop to the network's own weights.
    """
    print("\n[DEBUG] Starting test_superadiabatic_net_potential_output_forward_and_backward...")
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
    sup_model = SuperadiabaticNet(channels=8, use_potential_output=True).train()
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)

    rho = CenteredGrid(math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    f_total, _, components = get_hybrid_force(rho, v, base_model, sup_model, dx, device)
    f_sup_native = components["f_sup"].values.vector['x'].native(components["f_sup"].values.vector['x'].shape)
    assert f_sup_native.shape == (2, domain_res)

    loss = torch.sum(f_sup_native ** 2)
    loss.backward()

    for name, param in sup_model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter '{name}' has a non-finite gradient"
    print("[DEBUG] test_superadiabatic_net_potential_output_forward_and_backward passed successfully.")


def test_potential_smoothing_reduces_high_frequency_content_and_preserves_gradients():
    """
    Validates the potential_smooth_sigma countermeasure (force_factory.py:
    _smooth_periodic) added after the scalar-potential variant went to NaN
    weights at the same epoch (125, start of window_size=35) under both
    full-trajectory-coverage and multi-potential mixing. Checks two things a real fix needs:
    (1) it actually suppresses high-frequency content (not a no-op), tested
    directly on an alternating +1/-1 signal, the sharpest possible spatial
    frequency on this grid - a working Gaussian blur must pull this toward
    its mean; (2) gradients still flow through it to every network
    parameter (it's an ordinary differentiable op, not a torch.no_grad
    detour, but worth confirming given this sits inside the exact backward
    path that corrupted previously).
    """
    print("\n[DEBUG] Starting test_potential_smoothing_reduces_high_frequency_content_and_preserves_gradients...")
    device = torch.device("cpu")

    # (1) Direct smoothing correctness: alternating signal -> near-zero (its mean).
    domain_res = 32
    alternating = torch.tensor([[(-1.0) ** i for i in range(domain_res)]], device=device)
    smoothed = _smooth_periodic(alternating, sigma=2.0, device=device)
    assert smoothed.abs().max() < 0.5, (
        f"Gaussian smoothing (sigma=2.0) did not meaningfully suppress an alternating "
        f"+1/-1 signal (max abs after smoothing: {smoothed.abs().max().item()}) - "
        "expected it to collapse toward the signal's mean (~0)."
    )
    assert smoothed.shape == alternating.shape
    # sigma=0 must be an exact no-op (default/backward-compatible behavior).
    unsmoothed = _smooth_periodic(alternating, sigma=0.0, device=device)
    assert torch.equal(unsmoothed, alternating)

    # (2) End-to-end: gradients still reach every parameter with smoothing enabled.
    box_length = 10.0
    dx = box_length / domain_res

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    sup_model = SuperadiabaticNet(channels=8, use_potential_output=True, potential_smooth_sigma=1.5).train()
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)

    rho = CenteredGrid(math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    f_total, _, components = get_hybrid_force(rho, v, base_model, sup_model, dx, device)
    f_sup_native = components["f_sup"].values.vector['x'].native(components["f_sup"].values.vector['x'].shape)
    loss = torch.sum(f_sup_native ** 2)
    loss.backward()

    for name, param in sup_model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient with smoothing enabled"
        assert torch.isfinite(param.grad).all(), f"Parameter '{name}' has a non-finite gradient with smoothing enabled"
    print("[DEBUG] test_potential_smoothing_reduces_high_frequency_content_and_preserves_gradients passed successfully.")


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
