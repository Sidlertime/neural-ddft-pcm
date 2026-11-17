"""
Error-vs-time (RMSE, LJ units) companion to compare_3solvers_movie.py: for
each of the 3 comparison scenarios, plots density/velocity RMSE vs. true MD
for all 3 solvers (PhiFlow, Goddard, py-pde), saved into that scenario's own
results/report_figures/3solver_* directory alongside its movie.
"""
import glob
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Sibling checkout of the pseudo-spectral solver (github.com/Sidlertime/goddard-ddft);
# override with the GODDARD_ROOT environment variable.
GODDARD_ROOT = os.environ.get("GODDARD_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "goddard")))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAME_INTERVAL = 0.01

SCENARIOS = {
    "3solver_3a_equilibrium": {
        "md_path": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
        "title": "Potential 3a, equilibrium mode",
        "solvers": [
            ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_baseline_equilibrium_01", "PhiFlow", 4.0),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-08_ddft_neural_3a_compare_01"), "Goddard (pseudo-spectral)", 3.99),
            ("results/2026-11-12_pypde_smoketest_3a_01", "py-pde", 4.0),
        ],
    },
    "3solver_3a_hybrid": {
        "md_path": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
        "title": "Potential 3a, trained hybrid (vext) correction",
        "solvers": [
            ("results/2026-11-12_3solver_compare/3a_hybrid_vext_full_fixed_hybrid_01", "PhiFlow", 4.0),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-12_goddard_hybrid_smoketest_3a_03"), "Goddard (pseudo-spectral)", 3.99),
            ("results/2026-11-12_pypde_hybrid_smoketest_3a_01", "py-pde", 4.0),
        ],
    },
    "3solver_3d": {
        "md_path": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
        "title": "Potential 3d, zero-shot hybrid (vext) correction",
        "solvers": [
            ("results/2026-11-12_3solver_compare/3d_hybrid_zeroshot_vext_hybrid_02", "PhiFlow (zero-shot)", 4.0),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-12_goddard_3d_zeroshot_vext_01"), "Goddard (zero-shot)", 3.99),
            ("results/2026-11-12_pypde_3d_zeroshot_vext_01", "py-pde (zero-shot)", 4.0),
        ],
    },
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


def plot_scenario(scenario_name, scenario):
    rho_md, v_md = load_md_ground_truth(scenario["md_path"])
    fig, (ax_rho, ax_v) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for eval_dir, label, max_sim_time in scenario["solvers"]:
        if not os.path.isdir(eval_dir):
            print(f"  [skip - not found] {eval_dir}")
            continue
        t, rho_rmse, v_rmse = rmse_vs_time(eval_dir, rho_md, v_md, max_sim_time)
        ax_rho.plot(t, rho_rmse, label=label, lw=1.8)
        ax_v.plot(t, v_rmse, label=label, lw=1.8)

    ax_rho.set_ylabel(r"Density RMSE ($\rho\sigma^3$)")
    ax_rho.set_title(f"{scenario['title']}: solver error vs. MD")
    ax_rho.legend(fontsize=9)
    ax_rho.grid(True, linestyle=":", alpha=0.5)
    ax_v.set_ylabel(r"Velocity RMSE ($\sigma/\tau$)")
    ax_v.set_xlabel(r"Simulation time $t/\tau$")
    ax_v.legend(fontsize=9)
    ax_v.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = f"results/report_figures/{scenario_name}/error_vs_time.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.chdir(REPO_ROOT)
    for scenario_name, scenario in SCENARIOS.items():
        plot_scenario(scenario_name, scenario)


if __name__ == "__main__":
    main()
