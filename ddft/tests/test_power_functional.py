import os
import tempfile
import torch
import numpy as np
from phi.torch.flow import *
from neural_ddft.models.power_functional_net import PowerFunctionalNet
from neural_ddft.models.factory import _load_correction_checkpoint
from neural_ddft.physics.force_factory import get_hybrid_power_force


def test_power_functional_force_matches_finite_difference():
    """
    Validates the core new mechanism: f_sup = -(1/rho) * dP^ex/dv recovered
    via torch.autograd.grad w.r.t. the velocity field, against a numerical
    finite difference of the network's own scalar output P^ex. This is the
    same class of check that caught the round() shadowing bug in the
    scalar-field variant (test_pde_rollout.py) and the autograd-disconnection
    bug in the equilibrium path (test_force_factory.py) - a sign error or a
    wrong-variable-differentiated bug here would otherwise pass a
    loss-decreasing training run while being physically meaningless.

    Perturbing v at a single grid index also shifts grad_v at neighboring
    indices (grad_v is a central difference of v) - the finite-difference
    computation below recomputes grad_v consistently with get_hybrid_power_force's
    own torch.roll formula so both sides see the identical dependency.
    """
    print("\n[DEBUG] Starting test_power_functional_force_matches_finite_difference...")
    device = torch.device("cpu")
    domain_res = 32
    box_length = 10.0
    dx = box_length / domain_res

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    power_model = PowerFunctionalNet(channels=8).eval()
    # Zero-initialized output layer (by design) would make dP/dv trivially
    # zero everywhere - perturb it so there's a genuine, non-flat functional.
    with torch.no_grad():
        power_model.output_layer.weight.add_(0.3)

    rho_np = 0.8 + 0.1 * np.sin(2 * np.pi * np.arange(domain_res) / domain_res)
    v_np = 0.1 * np.cos(2 * np.pi * np.arange(domain_res) / domain_res)

    rho_grid = CenteredGrid(math.tensor(rho_np.astype(np.float32), spatial('x')),
                             extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v_grid = StaggeredGrid(math.tensor(v_np.astype(np.float32), spatial('x')),
                            extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    _, _, components = get_hybrid_power_force(rho_grid, v_grid, base_model, power_model, dx, device)
    f_sup_native = components["f_power"].values.vector['x'].native(components["f_power"].values.vector['x'].shape)
    f_sup_np = f_sup_native.detach().numpy().reshape(-1)
    # f_sup = -dP/dv / rho (rho comfortably above the safety clamp here) -
    # recover the autograd dP/dv to compare directly against the finite difference.
    autograd_dPdv = -f_sup_np * rho_np

    def p_exc_sum(rho_arr, v_arr):
        rho_pt = torch.tensor(rho_arr, dtype=torch.float32).unsqueeze(0)
        v_pt = torch.tensor(v_arr, dtype=torch.float32).unsqueeze(0)
        grad_rho_pt = (torch.roll(rho_pt, -1, dims=-1) - torch.roll(rho_pt, 1, dims=-1)) / (2 * dx)
        grad_v_pt = (torch.roll(v_pt, -1, dims=-1) - torch.roll(v_pt, 1, dims=-1)) / (2 * dx)
        with torch.no_grad():
            return power_model(rho_pt, v_pt, grad_rho_pt, grad_v_pt).item()

    eps = 1e-3
    rng = np.random.RandomState(0)
    sample_indices = rng.choice(domain_res, size=8, replace=False)

    for i in sample_indices:
        v_plus = v_np.copy()
        v_plus[i] += eps
        v_minus = v_np.copy()
        v_minus[i] -= eps

        fd_grad = (p_exc_sum(rho_np, v_plus) - p_exc_sum(rho_np, v_minus)) / (2 * eps)

        print(f"[DEBUG] index={i}: finite_diff={fd_grad:.6f}, autograd={autograd_dPdv[i]:.6f}")
        assert abs(fd_grad - autograd_dPdv[i]) < 1e-3 + 0.02 * abs(fd_grad), (
            f"Autograd dP^ex/dv at index {i} ({autograd_dPdv[i]}) disagrees with "
            f"finite difference ({fd_grad}) - possible sign error or autograd disconnection."
        )
    print("[DEBUG] test_power_functional_force_matches_finite_difference passed successfully.")


def test_power_functional_forward_and_backward_reaches_every_parameter():
    """
    Confirms gradients flow through every PowerFunctionalNet parameter when
    embedded in get_hybrid_power_force (not just the output layer - autograd
    could in principle silently disconnect an earlier layer), and that
    eval-mode (create_graph=False) doesn't error.
    """
    print("\n[DEBUG] Starting test_power_functional_forward_and_backward_reaches_every_parameter...")
    device = torch.device("cpu")
    domain_res = 32
    box_length = 10.0
    dx = box_length / domain_res

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    power_model = PowerFunctionalNet(
        channels=8, use_vext=True, use_rho_prev=True, use_v_prev=True,
        use_accel=True, use_gamma=True, use_eta=True,
    ).train()
    with torch.no_grad():
        power_model.output_layer.weight.add_(0.3)

    rho = CenteredGrid(math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    rho_prev = CenteredGrid(0.9 * math.ones(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v_prev = StaggeredGrid(math.zeros(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(math.random_normal(batch(batch=2), spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    f_total, _, components = get_hybrid_power_force(
        rho, v, base_model, power_model, dx, device,
        V_ext_grid=V_ext, rho_prev_grid=rho_prev, v_prev_grid=v_prev, dt=1e-4,
        gamma=0.1, eta=0.6944,
    )
    f_sup_native = components["f_power"].values.vector['x'].native(components["f_power"].values.vector['x'].shape)
    assert f_sup_native.shape == (2, domain_res)

    loss = torch.sum(f_sup_native ** 2)
    loss.backward()

    for name, param in power_model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter '{name}' has a non-finite gradient"

    # eval-mode (create_graph=False) path must also run cleanly.
    power_model.eval()
    get_hybrid_power_force(
        rho, v, base_model, power_model, dx, device,
        V_ext_grid=V_ext, rho_prev_grid=rho_prev, v_prev_grid=v_prev, dt=1e-4,
        gamma=0.1, eta=0.6944,
    )
    print("[DEBUG] test_power_functional_forward_and_backward_reaches_every_parameter passed successfully.")


def test_power_functional_zero_init_gives_near_zero_force():
    """
    A freshly-initialized PowerFunctionalNet (small-std-normal output
    weight, zero bias - same pattern as SuperadiabaticNet's output layer,
    see power_functional_net.py) must produce an f_sup close to zero, so
    training starts from (near-)pure equilibrium DDFT like every other
    correction network in this project.
    """
    print("\n[DEBUG] Starting test_power_functional_zero_init_gives_near_zero_force...")
    device = torch.device("cpu")
    domain_res = 32
    box_length = 10.0
    dx = box_length / domain_res

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        def forward(self, rho):
            return rho * 0.0 * self.weight

    base_model = MockBaseModel().eval()
    power_model = PowerFunctionalNet(channels=8).eval()  # untouched, as-initialized

    rho = CenteredGrid(0.8 + 0.1 * math.random_normal(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.1 * math.random_normal(spatial(x=domain_res)), extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    _, _, components = get_hybrid_power_force(rho, v, base_model, power_model, dx, device)
    f_sup_native = components["f_power"].values.vector['x'].native(components["f_power"].values.vector['x'].shape)
    assert f_sup_native.abs().max().item() < 1e-2, (
        f"Freshly-initialized PowerFunctionalNet produced a non-negligible force "
        f"(max abs {f_sup_native.abs().max().item()}) - expected near-zero at init."
    )
    print("[DEBUG] test_power_functional_zero_init_gives_near_zero_force passed successfully.")


def test_load_correction_checkpoint_rejects_power_functional_channel_mismatch():
    """
    Mirrors test_force_factory.py's use_potential_output mismatch check, for
    PowerFunctionalNet's own CONFIG_FLAGS (OPTIONAL_CHANNELS only - it has no
    output-interpretation flags since its output is always a scalar).
    """
    print("\n[DEBUG] Starting test_load_correction_checkpoint_rejects_power_functional_channel_mismatch...")
    trained_with_vext = PowerFunctionalNet(channels=4, use_vext=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "power.ckpt")
        torch.save({"state_dict": trained_with_vext.state_dict(), "use_vext": True}, ckpt_path)

        loaded_without_vext = PowerFunctionalNet(channels=4, use_vext=False)
        try:
            _load_correction_checkpoint(loaded_without_vext, ckpt_path, torch.device("cpu"))
            assert False, "Expected a ValueError for the use_vext mismatch"
        except ValueError as e:
            assert "use_vext" in str(e)

        # Matching flag loads cleanly.
        loaded_with_vext = PowerFunctionalNet(channels=4, use_vext=True)
        _load_correction_checkpoint(loaded_with_vext, ckpt_path, torch.device("cpu"))
    print("[DEBUG] test_load_correction_checkpoint_rejects_power_functional_channel_mismatch passed successfully.")
