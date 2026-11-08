"""
Derives a "PDE-consistent" velocity field from a potential's recorded
density trajectory alone, by inverting the continuity equation (the same
one pde_step/PhiFlowDDFT.step() integrate forward) instead of using MD's
directly-measured (and separately, independently noisy) velocity.

Why: MD's measured density and velocity are two independently-sampled
quantities. If the network trains against measured velocity, any mismatch
between "the velocity MD actually measured" and "the velocity that would
make the continuity equation exactly reproduce MD's own measured density"
is baked into the training target as an inconsistency the network can never
satisfy. This derives the latter as an alternative target - parameterized
by whatever diffusion_coeff production training settles on (see
continuity_diagnostic.py and configs/diffcoeff_stability_*), so the
inversion stays self-consistent with whichever integrator training
actually uses, not hardcoded to the zero-diffusion special case.

Method (single effective step per recorded 0.01-spaced frame pair -
continuity_diagnostic.py found sub-stepping granularity barely changes the
forward integrator's own fidelity, so this simpler route is justified for
the inversion too):
  1. Estimate d(rho)/dt via centered finite differences between frames.
  2. Integrate spatially: total_flux(x) = C - cumsum(d(rho)/dt)*dx, where
     total_flux = rho*v - diffusion_coeff*dx*|v|*grad(rho) is exactly what
     pde_step/PhiFlowDDFT.step() compute forward - the diffusive term is
     folded in self-consistently, not ignored.
  3. The free constant C is resolved by matching the derived v's spatial
     mean to the TRUE measured v's spatial mean at that frame (a zero-
     diffusion-based closed-form estimate for C, used as the anchor even
     when diffusion_coeff != 0 - matching the overall mean only needs to
     be approximately right, the continuity constraint is what determines
     the *shape*).
  4. v = total_flux / (rho -+ diffusion_coeff*dx*|v|*grad_rho) - this is
     implicit in v (the diffusive term depends on |v|, piecewise-linear in
     its sign) so a short fixed-point iteration solves it; diffusion_coeff=0
     collapses to a trivial direct division. Clamped against the same
     rho_floor used throughout this project to avoid the familiar 1/rho
     blowup in low-density edge regions.

Note: this uses a plain collocated (non-staggered) finite-difference
approximation of the continuity equation (np.roll-based central
differences), not PhiFlow's exact CenteredGrid/StaggeredGrid staggered
discretization that pde_step/PhiFlowDDFT.step() actually run - close enough
to be physically meaningful as a training target, but not a bit-exact
inverse of the production integrator's own grid convention. Verified
self-consistent on its own terms (forward-step-then-invert round-trip) by
tests/test_pde_velocity_derivation.py.

Usage:
    python scripts/tools/derive_pde_velocity.py \
        --md_path ../data/lj_data/ref_3a_langevin_01/avg_2000/ \
        --diffusion_coeff 0.0 \
        --output_dir results/diagnostics/pde_velocity_3a
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from neural_ddft.data.dataset import MDWindowDataset

RHO_FLOOR = 1e-6


def solve_velocity_from_flux(total_flux, rho_face, grad_rho_face, diffusion_coeff, dx, n_iters=6):
    """
    Inverts total_flux = rho_face*v - diffusion_coeff*dx*|v|*grad_rho_face
    for v. All three of total_flux/rho_face/grad_rho_face live on the same
    staggered (face) grid as v itself - see derive_velocity_frame. Implicit
    in v because of the |v| term - fixed-point iteration (trivial single
    division when diffusion_coeff == 0).
    """
    rho_face_safe = np.maximum(rho_face, RHO_FLOOR)
    v = total_flux / rho_face_safe  # initial guess, ignoring diffusion
    if diffusion_coeff == 0.0:
        return v
    c = diffusion_coeff * dx
    for _ in range(n_iters):
        denom = rho_face_safe - c * np.sign(v) * grad_rho_face
        denom = np.where(np.abs(denom) < RHO_FLOOR, np.sign(denom) * RHO_FLOOR + (denom == 0) * RHO_FLOOR, denom)
        v = total_flux / denom
    return v


def derive_velocity_frame(rho_t, drho_dt, v_true_t, dx, diffusion_coeff):
    """
    rho_t, drho_dt, v_true_t: (X,) arrays for one frame, on the density
    (cell-centered) grid.
    Returns v_derived: (X,) array, on the same staggered (face) grid v
    conventionally lives on.

    Staggered-grid convention (matches how the divergence of a face-valued
    flux maps back to a cell-centered quantity, the same structure
    PhiFlow's CenteredGrid/StaggeredGrid pde_step actually uses - critically
    NOT a central ([i+1]-[i-1]) difference, which does not telescope under
    a cumulative sum and silently breaks the inversion):
      - total_flux[i] lives on the face between cell i and cell i+1.
      - div_flux[i] (at cell center i) = (total_flux[i] - total_flux[i-1]) / dx
        => a cell-centered density update needs the face BEHIND it
           subtracted from the face AHEAD of it.
      - Inverting: total_flux[i] = total_flux[-1] - dx * cumsum(drho_dt)[i],
        which DOES telescope exactly from drho_dt[i] = -(total_flux[i] -
        total_flux[i-1]) / dx.
    """
    # Face-valued density/gradient, same location as total_flux and v.
    rho_face = 0.5 * (rho_t + np.roll(rho_t, -1))
    grad_rho_face = (np.roll(rho_t, -1) - rho_t) / dx

    raw_cumflux = -np.cumsum(drho_dt) * dx  # total_flux(x) - total_flux(-1), up to the free constant
    rho_face_safe = np.maximum(rho_face, RHO_FLOOR)

    # Resolve the free constant C via the zero-diffusion closed form:
    # v = (C + raw_cumflux) / rho_face  =>
    # mean(v) = C*mean(1/rho_face) + mean(raw_cumflux/rho_face)
    # Solve for C so mean(v) matches the true measured mean velocity.
    inv_rho_mean = np.mean(1.0 / rho_face_safe)
    raw_term_mean = np.mean(raw_cumflux / rho_face_safe)
    C = (np.mean(v_true_t) - raw_term_mean) / inv_rho_mean

    total_flux = C + raw_cumflux
    return solve_velocity_from_flux(total_flux, rho_face, grad_rho_face, diffusion_coeff, dx)


def derive_velocity_trajectory(rho_traj, v_traj, dx, frame_interval, diffusion_coeff):
    """
    rho_traj, v_traj: (T, X) arrays (full smoothed MD trajectory).
    Returns v_derived_traj: (T, X) - same shape, endpoints use one-sided
    differences for d(rho)/dt, interior frames use centered differences.
    """
    T = rho_traj.shape[0]
    drho_dt = np.empty_like(rho_traj)
    drho_dt[1:-1] = (rho_traj[2:] - rho_traj[:-2]) / (2 * frame_interval)
    drho_dt[0] = (rho_traj[1] - rho_traj[0]) / frame_interval
    drho_dt[-1] = (rho_traj[-1] - rho_traj[-2]) / frame_interval

    v_derived = np.empty_like(rho_traj)
    for t in range(T):
        v_derived[t] = derive_velocity_frame(rho_traj[t], drho_dt[t], v_traj[t], dx, diffusion_coeff)
    return v_derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md_path", required=True)
    parser.add_argument("--diffusion_coeff", type=float, default=0.0, help="Must match whatever diffusion_coeff production training settles on")
    parser.add_argument("--frame_interval", type=float, default=0.01)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = MDWindowDataset(args.md_path, window_size=1)
    rho_traj = ds.rho_traj.numpy()
    v_traj = ds.v_traj.numpy()
    dx = ds.dr
    T = rho_traj.shape[0]

    print(f"Loaded {T} frames, dx={dx:.5f}, diffusion_coeff={args.diffusion_coeff}")
    v_derived = derive_velocity_trajectory(rho_traj, v_traj, dx, args.frame_interval, args.diffusion_coeff)

    diff = v_derived - v_traj
    mse = np.mean(diff ** 2, axis=1)
    print(f"Derived-vs-true velocity MSE: mean={mse.mean():.6e}  max={mse.max():.6e}")
    true_scale = np.mean(v_traj ** 2)
    print(f"True velocity's own magnitude (mean v^2): {true_scale:.6e}")
    print(f"(derived-vs-true MSE as a fraction of true velocity's own scale: {mse.mean() / true_scale:.1%})")

    np.save(os.path.join(args.output_dir, "v_derived.npy"), v_derived.astype(np.float32))
    print(f"Saved derived velocity trajectory to {args.output_dir}/v_derived.npy")

    times = np.arange(T) * args.frame_interval
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, mse, lw=1.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Derived-vs-true velocity MSE")
    ax.set_title(f"PDE-derived vs. MD-measured velocity (diffusion_coeff={args.diffusion_coeff})\n{args.md_path}")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "velocity_derivation_error_vs_time.png"), dpi=200)
    plt.close(fig)

    sample_frames = sorted(set(int(f * (T - 1)) for f in [0.0, 0.25, 0.5, 0.75, 1.0]))
    fig, axes = plt.subplots(1, len(sample_frames), figsize=(4 * len(sample_frames), 3.5), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), sample_frames):
        ax.plot(v_traj[t], label="MD (measured)", lw=1.8)
        ax.plot(v_derived[t], label="derived", lw=1.4, linestyle="--")
        ax.set_title(f"frame {t} (t={t*args.frame_interval:.2f})")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes_flat = np.atleast_1d(axes)
    axes_flat[0].set_ylabel("Velocity")
    axes_flat[0].legend(fontsize=8)
    fig.suptitle(f"Sampled velocity profiles: PDE-derived vs. MD-measured\n{args.md_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "velocity_profile_samples.png"), dpi=200)
    plt.close(fig)

    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
