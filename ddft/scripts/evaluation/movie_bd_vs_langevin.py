"""
Movie version of plot_bd_vs_langevin.py's static comparison: animates the
raw (per-frame, not averaged) BD and Langevin density trajectories side by
side on the same axes, one movie per potential.
"""
import glob
import os
import sys

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from neural_ddft.data.dataset import MDWindowDataset  # noqa: E402

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


def main():
    out_dir = "results/report_figures/bd_vs_langevin"
    os.makedirs(out_dir, exist_ok=True)

    for name, paths in POTENTIALS.items():
        bd_files = sorted(glob.glob(f"{paths['bd_dir']}/collection.*.h5"))
        with h5py.File(bd_files[0], "r") as f:
            r_bd = np.array(f["r"]).flatten()
            V_bd = np.array(f["V"]).flatten()

        ds = MDWindowDataset(paths["langevin_dir"], window_size=1)
        rho_lang_traj = ds.rho_traj.numpy()
        r_lang = ds.r_np

        n_frames = min(len(bd_files), rho_lang_traj.shape[0])

        fig, ax = plt.subplots(figsize=(8, 5))
        ymax = max(np.array(h5py.File(bd_files[-1], "r")["n"]).max(), rho_lang_traj[-1].max())
        ax.set_xlim(0, max(r_bd[-1], r_lang[-1]))
        ax.set_ylim(0, ymax * 1.3)
        ax.set_xlabel(r"Position $z/\sigma$")
        ax.set_ylabel(r"Density $\rho\sigma^3$")
        ax.set_title(f"BD vs Langevin (raw trajectory) - {name}")

        with h5py.File(bd_files[0], "r") as f:
            rho0_bd = np.array(f["n"]).flatten()
        (line_bd,) = ax.plot(r_bd, rho0_bd, color="#1f77b4", lw=1.5, label="BD (overdamped)")
        (line_lang,) = ax.plot(r_lang, rho_lang_traj[0], color="#d62728", lw=1.5, ls="--", label="Langevin (underdamped)")
        ax.legend(loc="upper right")

        ax2 = ax.twinx()
        ax2.plot(r_lang, V_bd if len(V_bd) == len(r_lang) else np.interp(r_lang, r_bd, V_bd), color="0.7", lw=1.0)
        ax2.set_ylabel("External potential", color="0.5")
        ax2.tick_params(axis="y", labelcolor="0.5")

        def update(i):
            with h5py.File(bd_files[i], "r") as f:
                line_bd.set_ydata(np.array(f["n"]).flatten())
            line_lang.set_ydata(rho_lang_traj[i])
            return line_bd, line_lang

        anim = animation.FuncAnimation(fig, update, frames=range(n_frames), blit=True)
        anim.save(f"{out_dir}/bd_vs_langevin_movie_{name}.mp4", writer=animation.FFMpegWriter(fps=15))
        plt.close(fig)
        print(f"{name}: movie saved ({n_frames} frames)")


if __name__ == "__main__":
    main()
