"""
Diagnostic: does the continuity equation alone, driven by the TRUE MD
velocity trajectory, reproduce the MD's own density evolution?

Motivation: the PDE's continuity sub-step (d(rho)/dt = -div(rho*v)) has NO
dependence on gamma/eta/m/kBT/any internal force - those only appear in the
velocity (momentum) update, a separate equation (see
src/neural_ddft/physics/phiflow_pde.py:PhiFlowDDFT.step(), EQUATION 1 vs
EQUATION 2). This script isolates EQUATION 1 completely: starting from the
MD's own recorded density at each frame and using the MD's own recorded
(ground-truth) velocity - never a model, never a learned correction - it
checks whether the discretized integrator (including the artificial-
diffusion upwind stabilization term) reproduces the MD's own density at the
next frame. Any systematic error here is attributable to the PDE
discretization itself (dx/dt, the 0.5*dx*|v| diffusion coefficient),
independent of any physics-parameter choice or learned force - a question
that was open ("I sometimes have the feeling it dont") before this check.

Two step-size variants are run and reported side by side:
  - "substepped": matches the real simulation's actual integration dt
    (project default 1e-4), holding velocity fixed across the ~100
    sub-steps between two recorded MD frames (0.01 time units apart) -
    the most faithful reproduction of what a real rollout does.
  - "single_step": one big dt=frame_interval step directly - cruder, but
    isolates how much the artificial-diffusion term's cumulative effect
    depends on sub-stepping granularity versus being an inherent bias.

Usage:
    python scripts/tools/continuity_diagnostic.py \
        --md_path ../data/lj_data/ref_3a_langevin_01/avg_2000/ \
        --output_dir results/diagnostics/continuity_3a
"""
import os
import builtins
import argparse
import numpy as np
import matplotlib.pyplot as plt
from phi.torch.flow import *
# from phi.torch.flow import * shadows Python's builtin round() with PhiFlow's
# own (single-argument) round_ - see force_factory.py's identical fix and
# memory feedback_eval_template_flag_drift.md / scalar_potential_architecture.md
# for the history of this exact gotcha. Use builtins.round below.

from neural_ddft.data.dataset import MDWindowDataset


def continuity_step(rho_grid, v_grid, dt, dx, diffusion_coeff=0.5):
    """
    Mirrors phiflow_pde.py:PhiFlowDDFT.step()'s EQUATION 1 (continuity)
    exactly - kept in sync by hand since this is a standalone diagnostic,
    not a production code path. No gamma/eta/m/kBT/force term anywhere in
    here, by construction (same as the real implementation).

    diffusion_coeff: the project hardcodes this at 0.5 (D_art = 0.5*dx*|v|),
    a pure numerical-stabilization term with no physical basis (not part of
    the true continuity equation) - exposed here so its own contribution to
    the integrator error measured by this script can be isolated. 0.0
    reproduces the exact physical equation with no artificial diffusion at all.
    """
    rho_faces = rho_grid.at(v_grid)
    flux_centered = rho_faces * v_grid
    grad_rho = field.spatial_gradient(rho_grid)
    D_art = diffusion_coeff * dx * math.abs(v_grid)
    flux_diffusive = -D_art * grad_rho
    total_flux = flux_centered + flux_diffusive
    rho_next = rho_grid - dt * field.divergence(total_flux)
    return rho_next.with_values(math.maximum(rho_next.values, 1e-6))


def rollout_one_frame(rho_start, v_np, dt, n_substeps, domain_res, box_length, diffusion_coeff=0.5):
    v_grid = StaggeredGrid(
        math.tensor(v_np.astype(np.float32), spatial('x')),
        extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length),
    )
    rho_current = rho_start
    for _ in range(n_substeps):
        rho_current = continuity_step(rho_current, v_grid, dt, box_length / domain_res, diffusion_coeff)
    return rho_current


def run_variant(rho_traj, v_traj, dx, box_length, domain_res, dt, n_substeps, hold_velocity, diffusion_coeff=0.5):
    """
    hold_velocity: "start" (v[t], the only choice with a real physical
    rationale when sub-stepping with no finer-grained velocity available),
    "end" (v[t+1]), or "average" (0.5*(v[t]+v[t+1])) - a convenience to see
    how sensitive the result is to this choice.
    """
    T = rho_traj.shape[0]
    mse, mae, linf = [], [], []
    predicted = []
    for t in range(T - 1):
        rho_start = CenteredGrid(
            math.tensor(rho_traj[t].astype(np.float32), spatial('x')),
            extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length),
        )
        if hold_velocity == "start":
            v_np = v_traj[t]
        elif hold_velocity == "end":
            v_np = v_traj[t + 1]
        elif hold_velocity == "average":
            v_np = 0.5 * (v_traj[t] + v_traj[t + 1])
        else:
            raise ValueError(f"Unknown hold_velocity mode: {hold_velocity}")

        rho_pred_grid = rollout_one_frame(rho_start, v_np, dt, n_substeps, domain_res, box_length, diffusion_coeff)
        pred = rho_pred_grid.values.native('x').cpu().numpy()
        target = rho_traj[t + 1]
        diff = pred - target
        mse.append(float(np.mean(diff ** 2)))
        mae.append(float(np.mean(np.abs(diff))))
        linf.append(float(np.max(np.abs(diff))))
        predicted.append(pred)
    return {
        "mse": np.array(mse), "mae": np.array(mae), "linf": np.array(linf),
        "predicted": np.array(predicted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md_path", required=True, help="Path to one potential's MD directory (e.g. .../ref_3a_langevin_01/avg_2000/)")
    parser.add_argument("--dt", type=float, default=1e-4, help="Physical integration dt (project default 1e-4)")
    parser.add_argument("--frame_interval", type=float, default=0.01, help="Time between consecutive MD frames (project convention: save_interval=0.01)")
    parser.add_argument("--hold_velocity", choices=["start", "end", "average"], default="start")
    parser.add_argument("--diffusion_coeff", type=float, default=0.5, help="Artificial-diffusion coefficient (project default 0.5); 0.0 = exact physical continuity equation, no stabilization")
    parser.add_argument("--output_dir", default="results/diagnostics/continuity_check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # window_size is irrelevant here - only .rho_traj/.v_traj (the full,
    # Gaussian-smoothed trajectory, same smoothing the training/eval
    # pipeline treats as ground truth) are used, not the windowed __getitem__.
    ds = MDWindowDataset(args.md_path, window_size=1)
    rho_traj = ds.rho_traj.numpy()
    v_traj = ds.v_traj.numpy()
    dx = ds.dr
    box_length = ds.box_length
    domain_res = ds.domain_resolution
    n_substeps = max(1, builtins.round(args.frame_interval / args.dt))

    print(f"Loaded {rho_traj.shape[0]} frames, domain_res={domain_res}, box_length={box_length:.4f}, dx={dx:.5f}")
    print(f"Substepped variant: dt={args.dt}, {n_substeps} sub-steps per {args.frame_interval}-time-unit frame interval")
    print(f"Velocity held via: '{args.hold_velocity}', artificial-diffusion coefficient: {args.diffusion_coeff}\n")

    substepped = run_variant(rho_traj, v_traj, dx, box_length, domain_res,
                              dt=args.dt, n_substeps=n_substeps, hold_velocity=args.hold_velocity,
                              diffusion_coeff=args.diffusion_coeff)
    single_step = run_variant(rho_traj, v_traj, dx, box_length, domain_res,
                               dt=args.frame_interval, n_substeps=1, hold_velocity=args.hold_velocity,
                               diffusion_coeff=args.diffusion_coeff)

    for name, result in [("substepped (dt matches real sim)", substepped), ("single_step (crude, dt=frame_interval)", single_step)]:
        print(f"--- {name} ---")
        print(f"  Density MSE: mean={result['mse'].mean():.6e}  max={result['mse'].max():.6e}")
        print(f"  Density MAE: mean={result['mae'].mean():.6e}  max={result['mae'].max():.6e}")
        print(f"  Density Linf: mean={result['linf'].mean():.6e}  max={result['linf'].max():.6e}")

    # For reference, also report how large the MD's own frame-to-frame
    # density CHANGE is - without this, a small-looking MSE could still be
    # most of the actual signal (i.e. the integrator barely does anything
    # useful), or conversely look large while being negligible next to the
    # true dynamics' own magnitude.
    frame_to_frame_diff = rho_traj[1:] - rho_traj[:-1]
    reference_scale_mse = float(np.mean(frame_to_frame_diff ** 2))
    print(f"\nFor reference, MD's own frame-to-frame density-change MSE: {reference_scale_mse:.6e}")
    print("(if the integrator's error MSE above is comparable to or larger than this, the")
    print(" continuity step is not meaningfully reproducing the true dynamics)")

    # Plots: error vs time, and a few sampled profile comparisons.
    T = rho_traj.shape[0]
    times = np.arange(T - 1) * args.frame_interval

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, substepped["mse"], label="substepped (real sim dt)", lw=1.6)
    ax.plot(times, single_step["mse"], label="single_step (dt=frame_interval)", lw=1.6, linestyle="--")
    ax.axhline(reference_scale_mse, color="gray", linestyle=":", label="MD's own frame-to-frame change (reference scale)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density MSE (predicted vs. true next frame)")
    ax.set_title(f"Continuity-only integrator error vs. time (diffusion_coeff={args.diffusion_coeff})\n{args.md_path}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "continuity_error_vs_time.png"), dpi=200)
    plt.close(fig)

    sample_frames = sorted(set(int(f * (T - 2)) for f in [0.0, 0.25, 0.5, 0.75, 1.0]))
    fig, axes = plt.subplots(1, len(sample_frames), figsize=(4 * len(sample_frames), 3.5), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), sample_frames):
        ax.plot(rho_traj[t + 1], label="MD (true)", lw=1.8)
        ax.plot(substepped["predicted"][t], label="substepped", lw=1.4, linestyle="--")
        ax.plot(single_step["predicted"][t], label="single_step", lw=1.2, linestyle=":")
        ax.set_title(f"frame {t}->{t+1} (t={t*args.frame_interval:.2f})")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes_flat = np.atleast_1d(axes)
    axes_flat[0].set_ylabel("Density")
    axes_flat[0].legend(fontsize=8)
    fig.suptitle(f"Sampled density profiles: continuity-only prediction vs. MD ground truth\n{args.md_path}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "continuity_profile_samples.png"), dpi=200)
    plt.close(fig)

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
