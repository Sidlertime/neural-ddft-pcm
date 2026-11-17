"""
Report figure: artificial-diffusion-coefficient comparison.

Corrected structure (per user feedback 2026-11-12): the real comparison is
"same model_mode, same potential, varying diffusion_coeff only" - NOT model
vs model. Produces:
  - 4 individual panels: (hybrid, 3a), (hybrid, 3d), (equilibrium, 3a),
    (equilibrium, 3d) - each holds model_mode and potential fixed, varies
    only diffusion_coeff.
  - 2 combined panels: (3a, both model_modes overlaid), (3d, both model_modes
    overlaid) - i.e. panel 1+3 combined, and panel 2+4 combined. These are
    the only plots where model_mode varies within one axes.
Never 3a and 3d together on the same axes.
Density/velocity error reported as RMSE in LJ units throughout.
"""
import glob
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAME_INTERVAL = 0.01

# (eval_dir, diffusion_coeff, max_sim_time) per (model_mode, potential)
CELLS = {
    ("hybrid", "3a"): [
        ("results/2026-11-08_full_eval/diffcoeff0_3a_hybrid_01", 0.0, 4.0),
        ("results/2026-11-06_full_eval/v7a_3a_hybrid_01", 0.5, 4.0),
    ],
    ("hybrid", "3d"): [
        ("results/2026-11-08_full_eval/diffcoeff0_3d_hybrid_01", 0.0, 1.6526),
        ("results/2026-11-08_full_eval/diffcoeff025_3d_hybrid_01", 0.25, 1.8386),
        ("results/2026-11-07_full_eval/numerics_sweep_3d_dt_5e-5_full_hybrid_01", 0.5, 4.0),
    ],
    ("equilibrium", "3a"): [
        ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_diffcoeff0_equilibrium_01", 0.0, 4.0),
        ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_baseline_equilibrium_01", 0.5, 4.0),
    ],
    ("equilibrium", "3d"): [
        ("results/2026-11-12_diffcoeff_stability/3d_equilibrium_diffcoeff0_equilibrium_01", 0.0, 4.0),
        ("results/2026-11-12_diffcoeff_stability/3d_equilibrium_baseline_equilibrium_01", 0.5, 4.0),
    ],
}

MD_PATHS = {
    "3a": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
}


def load_md_ground_truth(md_path):
    files = sorted(glob.glob(os.path.join(md_path, "collection.*.h5")))
    rho_list, v_list = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            rho_list.append(np.array(h["n"]).flatten())
            v_list.append(np.array(h["vz"]).flatten())
    return np.stack(rho_list), np.stack(v_list)


def rmse_vs_time(eval_dir, rho_md, v_md, max_sim_time):
    rho_files = sorted(glob.glob(os.path.join(eval_dir, "rho*.npy")))
    times, rho_rmse, v_rmse = [], [], []
    for f in rho_files:
        t_str = os.path.basename(f)[3:-4]
        t = float(t_str)
        if t > max_sim_time + 1e-9:
            continue
        frame_idx = int(round(t / FRAME_INTERVAL))
        if frame_idx >= rho_md.shape[0]:
            continue
        rho_pred = np.load(f)
        v_pred = np.load(os.path.join(eval_dir, f"v{t_str}.npy"))
        times.append(t)
        rho_rmse.append(np.sqrt(np.mean((rho_pred - rho_md[frame_idx]) ** 2)))
        v_rmse.append(np.sqrt(np.mean((v_pred - v_md[frame_idx]) ** 2)))
    order = np.argsort(times)
    return np.array(times)[order], np.array(rho_rmse)[order], np.array(v_rmse)[order]


def _plot_series(ax_rho, ax_v, eval_dir, label, max_sim_time, rho_md, v_md, linestyle="-", color=None):
    if not os.path.isdir(eval_dir):
        print(f"  [skip - not found] {eval_dir}")
        return None
    t, rho_rmse, v_rmse = rmse_vs_time(eval_dir, rho_md, v_md, max_sim_time)
    line, = ax_rho.plot(t, rho_rmse, label=label, linestyle=linestyle, lw=1.8, color=color)
    ax_v.plot(t, v_rmse, label=label, linestyle=linestyle, lw=1.8, color=line.get_color())
    if max_sim_time < 3.99:
        ax_rho.axvline(max_sim_time, color=line.get_color(), alpha=0.4, lw=1, ls=":")
        ax_v.axvline(max_sim_time, color=line.get_color(), alpha=0.4, lw=1, ls=":")
    return line.get_color()


def plot_single_model(model_mode, potential, out_path):
    """Panel 1-4: fixed model_mode + potential, varying diffusion_coeff only."""
    rho_md, v_md = load_md_ground_truth(MD_PATHS[potential])
    fig, (ax_rho, ax_v) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for eval_dir, diffusion_coeff, max_sim_time in CELLS[(model_mode, potential)]:
        label = f"diffusion_coeff={diffusion_coeff}"
        _plot_series(ax_rho, ax_v, eval_dir, label, max_sim_time, rho_md, v_md)

    ax_rho.set_ylabel(r"Density RMSE ($\rho\sigma^3$)")
    ax_rho.set_title(f"{model_mode} model, potential {potential}: varying diffusion_coeff")
    ax_rho.legend(fontsize=9)
    ax_rho.grid(True, linestyle=":", alpha=0.5)
    ax_v.set_ylabel(r"Velocity RMSE ($\sigma/\tau$)")
    ax_v.set_xlabel(r"Simulation time $t/\tau$")
    ax_v.legend(fontsize=9)
    ax_v.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_potential(potential, out_path):
    """Panel 5-6: both model_modes overlaid, for one potential - the only
    plots where model_mode varies within one axes."""
    rho_md, v_md = load_md_ground_truth(MD_PATHS[potential])
    fig, (ax_rho, ax_v) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    linestyles = {"hybrid": "-", "equilibrium": "--"}
    for model_mode in ("hybrid", "equilibrium"):
        for eval_dir, diffusion_coeff, max_sim_time in CELLS[(model_mode, potential)]:
            label = f"{model_mode}, diffusion_coeff={diffusion_coeff}"
            _plot_series(ax_rho, ax_v, eval_dir, label, max_sim_time, rho_md, v_md, linestyle=linestyles[model_mode])

    ax_rho.set_ylabel(r"Density RMSE ($\rho\sigma^3$)")
    ax_rho.set_title(f"Potential {potential}: hybrid vs. equilibrium, all diffusion_coeff values")
    ax_rho.legend(fontsize=8)
    ax_rho.grid(True, linestyle=":", alpha=0.5)
    ax_v.set_ylabel(r"Velocity RMSE ($\sigma/\tau$)")
    ax_v.set_xlabel(r"Simulation time $t/\tau$")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("results/report_figures/diffcoeff_comparison_3a", exist_ok=True)
    os.makedirs("results/report_figures/diffcoeff_comparison_3d", exist_ok=True)

    # 4 individual panels
    plot_single_model("hybrid", "3a", "results/report_figures/diffcoeff_comparison_3a/diffcoeff_hybrid_3a.png")
    plot_single_model("hybrid", "3d", "results/report_figures/diffcoeff_comparison_3d/diffcoeff_hybrid_3d.png")
    plot_single_model("equilibrium", "3a", "results/report_figures/diffcoeff_comparison_3a/diffcoeff_equilibrium_3a.png")
    plot_single_model("equilibrium", "3d", "results/report_figures/diffcoeff_comparison_3d/diffcoeff_equilibrium_3d.png")

    # 2 combined panels
    plot_combined_potential("3a", "results/report_figures/diffcoeff_comparison_3a/diffcoeff_combined_3a.png")
    plot_combined_potential("3d", "results/report_figures/diffcoeff_comparison_3d/diffcoeff_combined_3d.png")


if __name__ == "__main__":
    main()
