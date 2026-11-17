"""
Compares the equilibrium (final-frame-averaged) density profile from the
newly-uploaded overdamped Brownian-Dynamics (BD) reference data against this
project's existing underdamped Langevin MD reference data, per potential.

Both datasets independently sample the SAME equilibrium distribution (same
free-energy functional + external potential, same box) via two different
dynamics methods starting from a near-uniform initial condition and relaxing
to a structured steady state over their respective trajectories - so if both
samplers are correct, their final-frame density profiles should agree
regardless of the very different relaxation dynamics along the way. This is
a cross-validation check, not a dynamics comparison.

One figure PER POTENTIAL (never combined - project convention), density
RMSE reported in reduced LJ units.
"""
import glob
import math
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from neural_ddft.data.dataset import MDWindowDataset  # noqa: E402

N_AVG = 20  # average the last N frames of each trajectory as the "equilibrium" estimate

POTENTIALS = {
    "gaussian": {
        "bd_dir": "../data/g1-closed-dz100/avg_5000",
        "langevin_dir": "../data/lj_data/ref_gaussian_langevin_01/avg_500/",
    },
    "3a": {
        "bd_dir": "../data/g28-closed-dz100/avg_5000",
        "langevin_dir": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    },
    "3b": {
        "bd_dir": "../data/g3-closed-dz100/avg_5000",
        "langevin_dir": "../data/lj_data/ref_3b_langevin_01/avg_2000/",
    },
    "3c": {
        "bd_dir": "../data/g4-closed-dz100/avg_5000",
        "langevin_dir": "../data/lj_data/ref_3c_langevin_01/avg_2000/",
    },
    "3d": {
        "bd_dir": "../data/g14-closed-dz100/avg_5000",
        "langevin_dir": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
    },
}


def load_bd_equilibrium(bd_dir):
    files = sorted(glob.glob(f"{bd_dir}/collection.*.h5"))
    assert len(files) > 0, f"No BD collection files found in {bd_dir}"
    rho_frames, V = None, None
    r = None
    rho_list = []
    for path in files[-N_AVG:]:
        with h5py.File(path, "r") as f:
            rho_list.append(np.array(f["n"]).flatten())
            if r is None:
                r = np.array(f["r"]).flatten()
                V = np.array(f["V"]).flatten()
    rho_eq = np.mean(rho_list, axis=0)
    return r, rho_eq, V, len(files)


def load_langevin_equilibrium(langevin_dir):
    ds = MDWindowDataset(langevin_dir, window_size=1)
    rho_traj = ds.rho_traj.numpy()
    rho_eq = rho_traj[-N_AVG:].mean(axis=0)
    r = ds.r_np
    V = ds.V_ext.numpy()
    return r, rho_eq, V, rho_traj.shape[0]


def main():
    out_dir = "results/report_figures/bd_vs_langevin"
    os.makedirs(out_dir, exist_ok=True)
    summary_lines = ["# BD vs Langevin equilibrium density comparison\n",
                      f"Both final-frame estimates average the last {N_AVG} saved frames of each trajectory.\n",
                      "| Potential | BD frames | Langevin frames | density RMSE (LJ units) |",
                      "|---|---|---|---|"]

    for name, paths in POTENTIALS.items():
        r_bd, rho_bd, V_bd, n_bd_frames = load_bd_equilibrium(paths["bd_dir"])
        r_lang, rho_lang, V_lang, n_lang_frames = load_langevin_equilibrium(paths["langevin_dir"])

        # Interpolate the finer BD grid (1000 pts) onto the coarser Langevin
        # grid (320 pts) for an RMSE number - both share the same box length.
        rho_bd_interp = np.interp(r_lang, r_bd, rho_bd, period=r_bd[-1] - r_bd[0] + (r_bd[1] - r_bd[0]))
        rmse = math.sqrt(np.mean((rho_bd_interp - rho_lang) ** 2))
        summary_lines.append(f"| {name} | {n_bd_frames} | {n_lang_frames} | {rmse:.4f} |")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r_bd, rho_bd, color="#1f77b4", lw=1.5, label=f"BD (overdamped), last {N_AVG} frames avg")
        ax.plot(r_lang, rho_lang, color="#d62728", lw=1.5, ls="--", label=f"Langevin (underdamped), last {N_AVG} frames avg")
        ax.set_xlabel(r"Position $z/\sigma$")
        ax.set_ylabel(r"Density $\rho\sigma^3$")
        ax.set_title(f"BD vs Langevin equilibrium density - {name} (RMSE={rmse:.4f})")
        ax.legend(loc="upper right")

        ax2 = ax.twinx()
        ax2.plot(r_lang, V_lang, color="0.7", lw=1.0, label="Vext")
        ax2.set_ylabel("External potential", color="0.5")
        ax2.tick_params(axis="y", labelcolor="0.5")

        fig.tight_layout()
        fig.savefig(f"{out_dir}/bd_vs_langevin_{name}.png", dpi=150)
        plt.close(fig)
        print(f"{name}: BD frames={n_bd_frames} langevin frames={n_lang_frames} density RMSE={rmse:.4f}")

    with open(f"{out_dir}/summary.md", "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote figures + summary.md to {out_dir}/")


if __name__ == "__main__":
    main()
