"""
Compares three model outputs per potential, each living in a different
physical framework but on the same box:
  - PCM:         genuine overdamped Smoluchowski DDFT solve on the BD/closed
                 grid (1000 pts, results/2026-11-13-bd_overdamped_test/<p>/),
                 reference = BD ground truth (../data/<bd_dir>/avg_5000).
  - equilibrium: this project's inertial PhiFlow solver, no learned
                 correction, on the Langevin grid (320 pts), reference =
                 Langevin MD ground truth.
  - hybrid:      same inertial solver WITH the best hybrid model (v7a,
                 single-potential-3a-trained, evaluated zero-shot elsewhere)
                 active - only where an existing eval run was found (reused,
                 not regenerated - some are t_range=1.5, some t_range=4.0,
                 noted per-potential).

Produces, per potential: one movie (density, PCM+equilibrium[+hybrid] vs
both MD ground truths) and an RMSE summary table. The 2-way (PCM vs
equilibrium) comparison covers all 5 potentials; the 3-way (+hybrid)
comparison only covers potentials where a hybrid eval already exists.
"""
import glob
import math
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from neural_ddft.data.dataset import MDWindowDataset  # noqa: E402

N_AVG = 20

POTENTIALS = {
    "gaussian": {
        "bd_dir": "../data/g1-closed-dz100/avg_5000",
        "pcm_dir": "results/2026-11-13-bd_overdamped_test/gaussian",
        "langevin_dir": "../data/lj_data/ref_gaussian_langevin_01/avg_500/",
        "equilibrium_dir": "results/2026-11-08_goddard_compare_phiflow_eq_equilibrium_01",
        "hybrid_dir": "results/2026-11-06_generalization_matrix/zeroshot_3a_to_gaussian_hybrid_01",
        "hybrid_t_range": 1.5,
    },
    "3a": {
        "bd_dir": "../data/g28-closed-dz100/avg_5000",
        "pcm_dir": "results/2026-11-13-bd_overdamped_test/3a",
        "langevin_dir": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
        "equilibrium_dir": "results/2026-11-12_diffcoeff_stability/3a_equilibrium_baseline_equilibrium_01",
        "hybrid_dir": "results/2026-11-06_full_eval/v7a_3a_hybrid_01",
        "hybrid_t_range": 4.0,
    },
    "3b": {
        "bd_dir": "../data/g3-closed-dz100/avg_5000",
        "pcm_dir": "results/2026-11-13-bd_overdamped_test/3b",
        "langevin_dir": "../data/lj_data/ref_3b_langevin_01/avg_2000/",
        "equilibrium_dir": "results/2026-11-13_diffcoeff_stability/3b_equilibrium_baseline_dt5e-5_equilibrium_01",
        "hybrid_dir": "results/2026-11-06_full_eval/3a_to_3b_hybrid_01",
        "hybrid_t_range": 4.0,
    },
    "3c": {
        "bd_dir": "../data/g4-closed-dz100/avg_5000",
        "pcm_dir": "results/2026-11-13-bd_overdamped_test/3c",
        "langevin_dir": "../data/lj_data/ref_3c_langevin_01/avg_2000/",
        "equilibrium_dir": "results/2026-11-13_diffcoeff_stability/3c_equilibrium_baseline_equilibrium_01",
        "hybrid_dir": "results/2026-11-06_generalization_matrix/zeroshot_3a_to_3c_hybrid_01",
        "hybrid_t_range": 1.5,
    },
    "3d": {
        "bd_dir": "../data/g14-closed-dz100/avg_5000",
        "pcm_dir": "results/2026-11-13-bd_overdamped_test/3d",
        "langevin_dir": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
        "equilibrium_dir": "results/2026-11-12_diffcoeff_stability/3d_equilibrium_baseline_equilibrium_01",
        "hybrid_dir": "results/single_potential_v7_v8/eval/2026-11-06_eval_v7a_zeroshot_on_3d_hybrid_01",
        "hybrid_t_range": 1.5,
    },
}


def load_rho_series(dir_path):
    files = sorted(glob.glob(f"{dir_path}/rho*.npy"),
                    key=lambda p: float(re.search(r"rho(-?\d+\.\d+)\.npy", p).group(1)))
    times = [float(re.search(r"rho(-?\d+\.\d+)\.npy", p).group(1)) for p in files]
    return times, files


def load_bd_ground_truth(bd_dir):
    files = sorted(glob.glob(f"{bd_dir}/collection.*.h5"))
    with h5py.File(files[0], "r") as f:
        r = np.array(f["r"]).flatten()
        V = np.array(f["V"]).flatten()
    rho_final = np.mean([np.array(h5py.File(f, "r")["n"]).flatten() for f in files[-N_AVG:]], axis=0)
    return r, rho_final, V


def load_langevin_ground_truth(langevin_dir):
    ds = MDWindowDataset(langevin_dir, window_size=1)
    rho_traj = ds.rho_traj.numpy()
    rho_final = rho_traj[-N_AVG:].mean(axis=0)
    return ds.r_np, rho_final, ds.V_ext.numpy()


def rmse(a, b, r_a, r_b):
    if len(a) != len(b):
        b_interp = np.interp(r_a, r_b, b, period=r_b[-1] - r_b[0] + (r_b[1] - r_b[0]))
        return math.sqrt(np.mean((a - b_interp) ** 2))
    return math.sqrt(np.mean((a - b) ** 2))


def main():
    out_dir = "results/report_figures/pcm_vs_equilibrium_vs_hybrid"
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for name, cfg in POTENTIALS.items():
        r_bd, rho_bd_final, V_bd = load_bd_ground_truth(cfg["bd_dir"])
        r_lang, rho_lang_final, V_lang = load_langevin_ground_truth(cfg["langevin_dir"])

        t_pcm, files_pcm = load_rho_series(cfg["pcm_dir"])
        t_eq, files_eq = load_rho_series(cfg["equilibrium_dir"])
        has_hybrid = os.path.isdir(cfg["hybrid_dir"])
        if has_hybrid:
            t_hy, files_hy = load_rho_series(cfg["hybrid_dir"])

        rho_pcm_final = np.load(files_pcm[-1])
        rho_eq_final = np.load(files_eq[-1])

        row = {
            "potential": name,
            "rmse_pcm_vs_bd": rmse(rho_pcm_final, rho_bd_final, r_bd, r_bd),
            "rmse_pcm_vs_langevin": rmse(rho_pcm_final, rho_lang_final, r_bd, r_lang),
            "rmse_eq_vs_langevin": rmse(rho_eq_final, rho_lang_final, r_lang, r_lang),
        }
        if has_hybrid:
            rho_hy_final = np.load(files_hy[-1])
            row["rmse_hybrid_vs_langevin"] = rmse(rho_hy_final, rho_lang_final, r_lang, r_lang)
            row["hybrid_t_range"] = cfg["hybrid_t_range"]
        summary_rows.append(row)

        # --- Movie ---
        n_frames = min(len(files_pcm), len(files_eq), len(files_hy) if has_hybrid else 10**9, 401)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, max(r_bd[-1], r_lang[-1]))
        ax.set_ylim(0, max(rho_bd_final.max(), rho_lang_final.max()) * 1.3)
        ax.set_xlabel(r"Position $z/\sigma$")
        ax.set_ylabel(r"Density $\rho\sigma^3$")
        ax.set_title(f"PCM vs equilibrium" + (" vs hybrid (v7a)" if has_hybrid else "") + f" - {name}")

        (line_bd,) = ax.plot(r_bd, np.load(files_pcm[0]) * 0 + rho_bd_final, color="0.6", lw=1, ls=":", label="BD ground truth")
        (line_lang,) = ax.plot(r_lang, rho_lang_final, color="0.3", lw=1, ls=":", label="Langevin MD ground truth")
        (line_pcm,) = ax.plot(r_bd, np.load(files_pcm[0]), color="#1f77b4", lw=1.5, label="PCM (overdamped)")
        (line_eq,) = ax.plot(r_lang, np.load(files_eq[0]), color="#d62728", lw=1.5, label="Equilibrium (inertial)")
        lines = [line_pcm, line_eq]
        if has_hybrid:
            (line_hy,) = ax.plot(r_lang, np.load(files_hy[0]), color="#2ca02c", lw=1.5, label="Hybrid v7a (inertial)")
            lines.append(line_hy)
        ax.legend(loc="upper right", fontsize=9)

        def update(i):
            line_pcm.set_ydata(np.load(files_pcm[min(i, len(files_pcm) - 1)]))
            line_eq.set_ydata(np.load(files_eq[min(i, len(files_eq) - 1)]))
            if has_hybrid:
                line_hy.set_ydata(np.load(files_hy[min(i, len(files_hy) - 1)]))
                return line_pcm, line_eq, line_hy
            return line_pcm, line_eq

        anim = animation.FuncAnimation(fig, update, frames=range(0, n_frames, 2), blit=True)
        anim.save(f"{out_dir}/pcm_vs_eq_vs_hybrid_{name}.mp4", writer=animation.FFMpegWriter(fps=15))
        plt.close(fig)
        print(f"{name}: done (n_frames={n_frames}, has_hybrid={has_hybrid})")

    with open(f"{out_dir}/summary.md", "w") as f:
        f.write("# PCM vs equilibrium vs hybrid - RMSE summary (final frame, density, LJ units)\n\n")
        f.write("| Potential | PCM vs BD GT | PCM vs Langevin MD GT | Equilibrium vs Langevin MD GT | Hybrid (v7a) vs Langevin MD GT | hybrid t_range |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in summary_rows:
            hy = f"{row.get('rmse_hybrid_vs_langevin', float('nan')):.4f}" if "rmse_hybrid_vs_langevin" in row else "n/a"
            tr = row.get("hybrid_t_range", "-")
            f.write(f"| {row['potential']} | {row['rmse_pcm_vs_bd']:.4f} | {row['rmse_pcm_vs_langevin']:.4f} | {row['rmse_eq_vs_langevin']:.4f} | {hy} | {tr} |\n")
    print(f"\nWrote movies + summary.md to {out_dir}/")


if __name__ == "__main__":
    main()
