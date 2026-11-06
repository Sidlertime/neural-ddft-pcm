import os
import tempfile
import torch
import numpy as np
import pytest
from phi.torch.flow import *
from neural_ddft.models.factory import build_models, _load_correction_checkpoint
from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.force_factory import get_equilibrium_force

CKPT_PATH = "models/c2_dz32.ckpt"
requires_checkpoint = pytest.mark.skipif(not os.path.exists(CKPT_PATH), reason="Base model checkpoint not found")


@requires_checkpoint
def test_equilibrium_chemical_potential_matches_finite_difference():
    """
    Validates the autograd half of the "double derivative" mechanism
    (chemical potential = dF/drho via PyTorch autograd, before PhiFlow's
    spatial_gradient turns it into a force) against a numerical finite
    difference of the base model's free energy output. This is the exact
    class of bug fixed by the "fix autograd disconnection in force factory"
    commit - a silently detached/zeroed gradient would pass a loss-decreasing
    training run while being physically wrong.
    """
    print("\n[DEBUG] Starting test_equilibrium_chemical_potential_matches_finite_difference...")
    device = torch.device("cpu")
    domain_res = 320
    box_length = 9.96875
    dx = box_length / domain_res

    models, _ = build_models(mode="equilibrium", device=device, dx=dx, trainable=False)
    base_model = models["base"]

    rho_np = 1.0 + 0.1 * np.sin(2 * np.pi * np.arange(domain_res) / domain_res)
    rho_grid = CenteredGrid(
        math.tensor(rho_np.astype(np.float32), spatial('x')),
        extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length)
    )

    _, dFdrho_grid = get_equilibrium_force(rho_grid, base_model, dx, device)
    # get_equilibrium_force divides the raw autograd gradient by dx; undo that
    # to compare directly against the finite difference of F_pred.sum().
    autograd_grad = (dFdrho_grid.values.native(dFdrho_grid.values.shape) * dx).detach().numpy().reshape(-1)

    eps = 1e-3
    rho_pt = torch.tensor(rho_np, dtype=torch.float32)
    rng = np.random.RandomState(0)
    sample_indices = rng.choice(domain_res, size=8, replace=False)

    with torch.no_grad():
        for i in sample_indices:
            rho_plus = rho_pt.clone()
            rho_plus[i] += eps
            rho_minus = rho_pt.clone()
            rho_minus[i] -= eps

            F_plus = base_model(rho_plus.unsqueeze(0).unsqueeze(1)).sum().item()
            F_minus = base_model(rho_minus.unsqueeze(0).unsqueeze(1)).sum().item()
            fd_grad = (F_plus - F_minus) / (2 * eps)

            print(f"[DEBUG] index={i}: finite_diff={fd_grad:.6f}, autograd={autograd_grad[i]:.6f}")
            assert abs(fd_grad - autograd_grad[i]) < 1e-3 + 0.02 * abs(fd_grad), (
                f"Autograd dF/drho at index {i} ({autograd_grad[i]}) disagrees with "
                f"finite difference ({fd_grad}) - possible autograd disconnection."
            )
    print("[DEBUG] test_equilibrium_chemical_potential_matches_finite_difference passed successfully.")


def test_load_correction_checkpoint_rejects_potential_output_mismatch():
    """
    A checkpoint trained with use_potential_output=True reinterprets the
    exact same weight shapes as a different physical quantity (a potential
    field vs. a force field) than use_potential_output=False - loading one
    into the other must raise, not silently misinterpret the weights, same
    as the existing OPTIONAL_CHANNELS mismatch check this extends.
    """
    print("\n[DEBUG] Starting test_load_correction_checkpoint_rejects_potential_output_mismatch...")
    trained_as_potential = SuperadiabaticNet(channels=4, use_potential_output=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "sup.ckpt")
        torch.save(
            {"state_dict": trained_as_potential.state_dict(), "use_potential_output": True},
            ckpt_path,
        )

        loaded_as_force = SuperadiabaticNet(channels=4, use_potential_output=False)
        try:
            _load_correction_checkpoint(loaded_as_force, ckpt_path, torch.device("cpu"))
            assert False, "Expected a ValueError for the use_potential_output mismatch"
        except ValueError as e:
            assert "use_potential_output" in str(e)

        # Matching flag loads cleanly.
        loaded_as_potential = SuperadiabaticNet(channels=4, use_potential_output=True)
        _load_correction_checkpoint(loaded_as_potential, ckpt_path, torch.device("cpu"))
    print("[DEBUG] test_load_correction_checkpoint_rejects_potential_output_mismatch passed successfully.")
