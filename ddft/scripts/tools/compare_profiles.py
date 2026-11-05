import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from neural_ddft.utils.metrics import (
    load_npy_series, load_md_series, compute_metrics,
    evaluate_single_prediction, summarize_run,
)


def plot_comparisons(reports, labels, output_dir):
    """
    Generates comparison plots across multiple prediction runs vs reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    first_rep = reports[0]
    times = first_rep["times"]
    n_frames = len(times)

    # 1. Error vs Time Plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for rep, label in zip(reports, labels):
        rho_mse_time = [m["mse"] for m in rep["per_step_rho"]]
        axes[0].plot(times[:len(rho_mse_time)], rho_mse_time, label=label, lw=1.8)
        if "per_step_v" in rep:
            v_mse_time = [m["mse"] for m in rep["per_step_v"]]
            axes[1].plot(times[:len(v_mse_time)], v_mse_time, label=label, lw=1.8)

    axes[0].set_ylabel("Density MSE")
    axes[0].set_title("Density Error Evolution Over Time")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_ylabel("Velocity MSE")
    axes[1].set_xlabel("Time Step / Frame")
    axes[1].set_title("Velocity Error Evolution Over Time")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, "error_vs_time.png")
    fig.savefig(error_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved error evolution plot to: {error_plot_path}")

    # 2. Spatial Profile Snapshots (Early, Mid, Late)
    snapshot_indices = [
        0,
        n_frames // 4,
        n_frames // 2,
        max(n_frames - 1, 0)
    ]
    snapshot_indices = sorted(list(set(snapshot_indices)))

    fig, axes = plt.subplots(len(snapshot_indices), 2, figsize=(12, 3 * len(snapshot_indices)), sharex=True)
    if len(snapshot_indices) == 1:
        axes = np.expand_dims(axes, 0)

    r_coord = first_rep["r_grid"]
    if r_coord is None:
        r_coord = np.arange(first_rep["t_rho"].shape[1])

    for row_idx, frame_idx in enumerate(snapshot_indices):
        t_val = times[frame_idx]

        # Density column
        axes[row_idx, 0].plot(r_coord, first_rep["t_rho"][frame_idx], "k--", label="Reference", lw=2)
        for rep, label in zip(reports, labels):
            idx = min(frame_idx, len(rep["p_rho"])-1)
            axes[row_idx, 0].plot(r_coord, rep["p_rho"][idx], label=label, alpha=0.85)
        axes[row_idx, 0].set_ylabel(f"t={t_val:.3f}\nDensity $\\rho$")
        axes[row_idx, 0].grid(True, linestyle=":", alpha=0.6)
        if row_idx == 0:
            axes[row_idx, 0].set_title("Density Profile $\\rho(r)$")
            axes[row_idx, 0].legend()

        # Velocity column
        if first_rep["t_v"] is not None:
            axes[row_idx, 1].plot(r_coord, first_rep["t_v"][frame_idx], "k--", label="Reference", lw=2)
            for rep, label in zip(reports, labels):
                if rep["p_v"] is not None:
                    idx = min(frame_idx, len(rep["p_v"])-1)
                    axes[row_idx, 1].plot(r_coord, rep["p_v"][idx], label=label, alpha=0.85)
            axes[row_idx, 1].set_ylabel(f"Velocity $v_z$")
            axes[row_idx, 1].grid(True, linestyle=":", alpha=0.6)
            if row_idx == 0:
                axes[row_idx, 1].set_title("Velocity Profile $v_z(r)$")

    axes[-1, 0].set_xlabel("Spatial coordinate $r$")
    axes[-1, 1].set_xlabel("Spatial coordinate $r$")
    plt.tight_layout()
    profiles_plot_path = os.path.join(output_dir, "profile_snapshots.png")
    fig.savefig(profiles_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved snapshot profiles plot to: {profiles_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare density & velocity error across multiple runs.")
    parser.add_argument("--pred", type=str, nargs="+", required=True, help="One or more prediction folders (.npy files)")
    parser.add_argument("--labels", type=str, nargs="+", default=None, help="Optional labels for each prediction folder")
    parser.add_argument("--ref", type=str, required=True, help="Reference folder (.npy files) or MD folder (.h5 files)")
    parser.add_argument("--is_md", action="store_true", help="Set this flag if --ref points to an MD collection folder with .h5 files")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional limit on number of time frames evaluated")
    parser.add_argument("--plot", action="store_true", help="Generate comparative plots for profiles and errors")
    parser.add_argument("--plot_dir", type=str, default="results/comparison_plots", help="Directory to save generated plots")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write overall JSON summary")
    args = parser.parse_args()

    pred_dirs = args.pred
    labels = args.labels if args.labels and len(args.labels) == len(pred_dirs) else [os.path.basename(os.path.normpath(p)) for p in pred_dirs]

    reports = []
    print("\n" + "=" * 78)
    print("                      DDFT MODEL COMPARISON SUMMARY")
    print("=" * 78)
    print(f"Reference: {args.ref} (is_md={args.is_md})\n")

    header = f"{'Model / Run Label':<28} | {'Rho MSE':<11} | {'Rho Linf':<11} | {'Vel MSE':<11} | {'Vel Linf':<11}"
    print(header)
    print("-" * len(header))

    json_summary = {}

    for pred_dir, label in zip(pred_dirs, labels):
        rep = evaluate_single_prediction(pred_dir, args.ref, is_ref_md=args.is_md, max_steps=args.max_steps)
        reports.append(rep)

        rho_mse = rep["density_metrics"]["overall_mse"]
        rho_linf = rep["density_metrics"]["overall_linf"]
        has_v = "velocity_metrics" in rep
        v_mse_str = f"{rep['velocity_metrics']['overall_mse']:.5e}" if has_v else "N/A"
        v_linf_str = f"{rep['velocity_metrics']['overall_linf']:.5e}" if has_v else "N/A"

        print(f"{label:<28} | {rho_mse:.5e} | {rho_linf:.5e} | {v_mse_str:<11} | {v_linf_str:<11}")

        json_summary[label] = {
            "pred_path": pred_dir,
            "density_metrics": rep["density_metrics"],
            "velocity_metrics": rep.get("velocity_metrics", None)
        }

    print("=" * 78 + "\n")

    if args.plot:
        plot_comparisons(reports, labels, args.plot_dir)

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(json_summary, fp, indent=2)
        print(f"Comparison summary saved to {args.out}")


if __name__ == "__main__":
    main()
