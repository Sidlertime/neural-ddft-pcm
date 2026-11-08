import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tools"))
from derive_pde_velocity import derive_velocity_frame, solve_velocity_from_flux  # noqa: E402


def forward_continuity_step(rho, v, dx, dt, diffusion_coeff):
    """
    Mirrors derive_velocity_frame's own assumed staggered-grid
    discretization exactly (face-valued total_flux, adjacent-difference
    divergence - see that function's docstring for why a central
    ([i+1]-[i-1]) difference would NOT telescope correctly under cumsum)
    - used here to generate a self-consistent (rho0, rho1, v) triple for
    the round-trip correctness check below.
    """
    rho_face = 0.5 * (rho + np.roll(rho, -1))
    grad_rho_face = (np.roll(rho, -1) - rho) / dx
    D_art = diffusion_coeff * dx * np.abs(v)
    total_flux = rho_face * v - D_art * grad_rho_face
    div_flux = (total_flux - np.roll(total_flux, 1)) / dx
    return rho - dt * div_flux


def _run_roundtrip_check(diffusion_coeff):
    rng = np.random.RandomState(0)
    X = 64
    dx = 10.0 / X
    dt = 0.01

    x = np.arange(X) * dx
    rho0 = 0.6 + 0.15 * np.sin(2 * np.pi * x / 10.0) + 0.05 * np.sin(4 * np.pi * x / 10.0 + 0.7)
    v_true = 0.05 * np.cos(2 * np.pi * x / 10.0 + 0.3) + 0.02 * rng.randn(X)

    rho1 = forward_continuity_step(rho0, v_true, dx, dt, diffusion_coeff)
    drho_dt = (rho1 - rho0) / dt

    v_recovered = derive_velocity_frame(rho0, drho_dt, v_true, dx, diffusion_coeff)

    max_abs_err = np.max(np.abs(v_recovered - v_true))
    print(f"[DEBUG] diffusion_coeff={diffusion_coeff}: max |v_recovered - v_true| = {max_abs_err:.3e}")
    assert max_abs_err < 1e-4, (
        f"Round-trip (forward continuity step then invert) failed to recover the true velocity "
        f"at diffusion_coeff={diffusion_coeff}: max abs error {max_abs_err}"
    )


def test_velocity_derivation_roundtrip_zero_diffusion():
    """
    Primary expected case (diffusion_coeff=0, the "most physical" setting
    per the user's own framing): a direct division, should recover the true
    velocity to near machine precision.
    """
    _run_roundtrip_check(diffusion_coeff=0.0)


def test_velocity_derivation_roundtrip_nonzero_diffusion():
    """
    Fallback case if diffusion_coeff=0 training turns out unstable and a
    small nonzero value is used instead - exercises the fixed-point
    iteration in solve_velocity_from_flux, not just the trivial division.
    """
    _run_roundtrip_check(diffusion_coeff=0.3)


def test_solve_velocity_from_flux_matches_direct_division_at_zero_diffusion():
    """Sanity check the function used standalone, not just via derive_velocity_frame."""
    rng = np.random.RandomState(1)
    total_flux = rng.randn(32)
    rho = 0.5 + 0.3 * rng.rand(32)
    grad_rho = rng.randn(32) * 0.1
    v = solve_velocity_from_flux(total_flux, rho, grad_rho, diffusion_coeff=0.0, dx=0.1)
    assert np.allclose(v, total_flux / rho)
