"""
Report-style figures for evaluation runs.

A "run" is an evaluation folder written by scripts/evaluation/phiflow_eval.py
(rho*.npy, v*.npy, and for hybrid runs f_eq*.npy / f_sup*.npy per saved frame).
Every function takes the folders as arguments and saves a .pdf and a .png.

    from neural_ddft.utils import plotting
    plotting.rmse_vs_time([(baseline_dir, "PCM only"), (model_dir, "Hybrid")], md_dir, "out/rmse")
"""
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_ddft.utils.metrics import evaluate_single_prediction

# colors (categorical order) and neutrals
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
INK, GRID, ZERO = "#2b2b2b", "#e4e4e0", "#b0b0aa"
TIME_SHADES = ["#9ec5f4", "#3987e5", "#104281"]   # light -> dark for increasing time

STYLE = {
    "font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
}
LABEL_Z, LABEL_T = r"$z$ [$\sigma$]", r"$t$ [$\tau$]"
LABEL_RHO, LABEL_V = r"$\rho$ [$\sigma^{-3}$]", r"$v$ [$\sigma/\tau$]"


def apply_style():
    plt.rcParams.update(STYLE)


def save(fig, out, dpi=300):
    """Save `out`.pdf and `out`.png (creates the folder)."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=dpi)
    plt.close(fig)
    print(f"saved {out}.pdf/.png")


def _colors(runs):
    """runs: list of (dir, label) or (dir, label, color); fills missing colors in palette order."""
    return [(r[0], r[1], r[2] if len(r) > 2 else PALETTE[i % len(PALETTE)]) for i, r in enumerate(runs)]


def _grid(ax, xlim=(0, 10), xticks=(0, 5, 10)):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)


def _top_legend(fig, ax, ncol, fontsize=7, top=0.86):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, fontsize=fontsize, bbox_to_anchor=(0.5, 1.0))
    return (0, 0, 1, top)


def _load_field(run_dir, prefix, t):
    return np.load(os.path.join(run_dir, f"{prefix}{t:06.3f}.npy")).ravel()


# ---------------------------------------------------------------------------
# Error and profiles
# ---------------------------------------------------------------------------

def rmse_vs_time(runs, md_dir, out, t_max=4.0):
    """Density and velocity RMSE vs. MD over time, one line per run."""
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.75), sharex=True)
    for path, label, color in _colors(runs):
        r = evaluate_single_prediction(path, md_dir, is_ref_md=True)
        t = r["times"]
        ax[0].plot(t, np.sqrt([m["mse"] for m in r["per_step_rho"]]), color=color, lw=1.2, label=label)
        ax[1].plot(t, np.sqrt([m["mse"] for m in r["per_step_v"]]), color=color, lw=1.2, label=label)
    for a, yl in zip(ax, [r"RMSE $\rho$ [$\sigma^{-3}$]", r"RMSE $v$ [$\sigma/\tau$]"]):
        a.set_xlabel(LABEL_T); a.set_ylabel(yl); a.grid(True, color=GRID, lw=0.5)
        a.set_xlim(0, t_max); a.set_ylim(bottom=0)
    rect = _top_legend(fig, ax[0], ncol=len(runs))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=rect)
    save(fig, out)


def density_rmse_panels(panels, out):
    """
    Density RMSE over time in several panels, e.g. the artificial-diffusion comparison.
    panels: list of (title, md_dir, runs). A run that stopped early is marked with x.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 1.9))
    axes = np.atleast_1d(axes)
    for a, (title, md_dir, runs) in zip(axes, panels):
        for path, label, color in _colors(runs):
            r = evaluate_single_prediction(path, md_dir, is_ref_md=True)
            er = np.sqrt([m["mse"] for m in r["per_step_rho"]])
            t = r["times"][:len(er)]
            a.plot(t, er, color=color, lw=1.2, label=label)
            if t[-1] < 3.9:
                a.plot(t[-1], er[-1], "x", color=color, ms=6, mew=1.5)
        a.set_title(title, fontsize=8); a.set_xlim(0, 4); a.set_ylim(bottom=0)
        a.grid(True, color=GRID, lw=0.5); a.set_xlabel(LABEL_T)
    axes[0].set_ylabel(r"RMSE $\rho$ [$\sigma^{-3}$]")
    rect = _top_legend(fig, axes[0], ncol=len(panels[0][2]), top=0.85)
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=rect)
    save(fig, out)


def profile_snapshots(runs, md_dir, out, times=(0.5, 1.0, 2.0, 4.0)):
    """Density and velocity profiles of MD and the runs at selected times."""
    apply_style()
    reps = [(evaluate_single_prediction(p, md_dir, is_ref_md=True), lab, c) for p, lab, c in _colors(runs)]
    r0 = reps[0][0]
    t_all = r0["times"]
    z = r0["r_grid"] if r0["r_grid"] is not None else np.arange(r0["t_rho"].shape[1]) / 32
    fig, ax = plt.subplots(len(times), 2, figsize=(6.4, 1.4 * len(times)), sharex=True, sharey="col")
    for i, ts in enumerate(times):
        k = int(np.argmin(np.abs(t_all - ts)))
        ax[i, 0].plot(z, r0["t_rho"][k], color=INK, lw=1.0, ls="--", label="MD")
        ax[i, 1].plot(z, r0["t_v"][k], color=INK, lw=1.0, ls="--", label="MD")
        for rep, lab, c in reps:
            ax[i, 0].plot(z, rep["p_rho"][k], color=c, lw=1.2, label=lab)
            ax[i, 1].plot(z, rep["p_v"][k], color=c, lw=1.2, label=lab)
        ax[i, 0].set_ylabel(f"$t={t_all[k]:.1f}\\tau$\n" + LABEL_RHO)
        ax[i, 1].set_ylabel(LABEL_V)
        for a in ax[i]:
            _grid(a, xticks=(0, 2.5, 5, 7.5, 10))
    for a in ax[-1]:
        a.set_xlabel(LABEL_Z)
    rect = _top_legend(fig, ax[0, 0], ncol=len(runs) + 1, fontsize=8, top=0.96)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=1.0, rect=rect)
    save(fig, out)


def potentials_overview(potentials, out, frames=((0, "$t=0$"), (99, r"$t=1\tau$"), (399, r"$t=4\tau$"))):
    """External potential (top) and MD density at selected frames (bottom) per potential.
    potentials: list of (name, md_dir)."""
    apply_style()
    fig, ax = plt.subplots(2, len(potentials), figsize=(1.725 * len(potentials), 2.6), sharex=True, sharey="row")
    for j, (name, md_dir) in enumerate(potentials):
        files = sorted(glob.glob(os.path.join(md_dir, "collection.*.h5")))
        with h5py.File(files[0]) as f:
            z, V = np.ravel(f["r"]), np.ravel(f["V"])
        ax[0, j].plot(z, V, color=INK, lw=1.2)
        ax[0, j].set_title(name, fontsize=9)
        for (i, label), color in zip(frames, TIME_SHADES):
            with h5py.File(files[i]) as f:
                ax[1, j].plot(z, np.ravel(f["n"]), color=color, lw=1.0, label=label)
        for a in ax[:, j]:
            _grid(a)
        ax[1, j].set_xlabel(LABEL_Z)
    ax[0, 0].set_ylabel(r"$V_\mathrm{ext}$ [$\epsilon$]")
    ax[1, 0].set_ylabel(LABEL_RHO + " (MD)")
    ax[1, min(1, len(potentials) - 1)].legend(loc="lower center", fontsize=6.5, ncol=len(frames),
                                              columnspacing=0.5, handlelength=0.9, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.4)
    save(fig, out)


# ---------------------------------------------------------------------------
# Adiabatic / superadiabatic split of a hybrid run
# ---------------------------------------------------------------------------

def _currents(model_dir, baseline_dir, md_dir):
    rep_h = evaluate_single_prediction(model_dir, md_dir, is_ref_md=True)
    rep_b = evaluate_single_prediction(baseline_dir, md_dir, is_ref_md=True)
    n = min(len(rep_h["times"]), len(rep_b["times"]))
    J_md = rep_h["t_rho"][:n] * rep_h["t_v"][:n]
    J_ad = rep_b["p_rho"][:n] * rep_b["p_v"][:n]
    J = rep_h["p_rho"][:n] * rep_h["p_v"][:n]
    return rep_h["times"][:n], rep_h["p_rho"][:n], J_md, J_ad, J


def flux_split_times(model_dir, baseline_dir, md_dir, out, times=(0.5, 1.0, 2.0, 4.0)):
    """
    Current J = rho*v at fixed times: MD, PCM-only (adiabatic, J_ad), hybrid (J)
    and the superadiabatic part J_sup = J - J_ad.
    """
    apply_style()
    T, _, J_md, J_ad, J = _currents(model_dir, baseline_dir, md_dir)
    z = np.arange(J.shape[1]) / 32
    fig, ax = plt.subplots(1, len(times), figsize=(6.9, 2.0), sharey=True)
    for a, ts in zip(ax, times):
        k = int(np.argmin(np.abs(T - ts)))
        a.axhline(0, color=ZERO, lw=0.5)
        a.plot(z, J_md[k], color=INK, lw=0.9, ls="--", label="MD")
        a.plot(z, J_ad[k], color=ORANGE, lw=1.1, label=r"PCM only $J_\mathrm{ad}$")
        a.plot(z, J[k], color=BLUE, lw=1.1, label=r"hybrid $J$")
        a.plot(z, J[k] - J_ad[k], color=AQUA, lw=1.1, label=r"superadiabatic $J_\mathrm{sup}=J-J_\mathrm{ad}$")
        a.set_title(f"$t={T[k]:.1f}\\tau$", fontsize=8)
        _grid(a)
        a.set_xlabel(LABEL_Z)
    ax[0].set_ylabel(r"$J=\rho v$ [$\sigma^{-2}\tau^{-1}$]")
    rect = _top_legend(fig, ax[0], ncol=4)
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=rect)
    save(fig, out)


def flux_split_maps(model_dir, baseline_dir, md_dir, out):
    """Space-time maps of the currents (MD, PCM-only, hybrid, J_sup) and of the force densities."""
    apply_style()
    T, rho, J_md, J_ad, J = _currents(model_dir, baseline_dir, md_dir)
    f_eq = np.array([_load_field(model_dir, "f_eq", t) for t in T])
    f_sup = np.array([_load_field(model_dir, "f_sup", t) for t in T])
    F_ad, F_sup = rho * f_eq, rho * f_sup
    extent = [0, 10, T[0], T[-1]]
    vJ, vD, vF = np.percentile(np.abs(J_md), 99.5), np.abs(J - J_ad).max(), np.percentile(np.abs(F_ad), 99.5)
    panels = [(J_md, vJ, r"MD current $J=\rho v$"), (J_ad, vJ, r"PCM only: $J_\mathrm{ad}$"), (J, vJ, r"Hybrid: $J$"),
              (J - J_ad, vD, r"$J_\mathrm{sup}=J_\mathrm{hybrid}-J_\mathrm{PCM}$"),
              (F_ad, vF, r"adiabatic $\rho f_\mathrm{eq}$"), (F_sup, vF, r"superadiabatic $\rho f_\mathrm{sup}$")]
    fig, ax = plt.subplots(2, 3, figsize=(6.9, 4.3), sharex=True, sharey=True)
    for a, (data, vmax, title) in zip(ax.ravel(), panels):
        im = a.imshow(data, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
        a.set_title(title, fontsize=8)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
        a.set_xticks([0, 5, 10])
    for a in ax[:, 0]:
        a.set_ylabel(LABEL_T)
    for a in ax[-1]:
        a.set_xlabel(LABEL_Z)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    save(fig, out)


def internal_force_split(model_dir, out, times=(0.5, 1.0, 2.0, 4.0)):
    """Adiabatic f_eq, superadiabatic f_sup and their sum of a hybrid run at fixed times."""
    apply_style()
    fig, ax = plt.subplots(1, len(times), figsize=(6.9, 1.9), sharey=True)
    for a, ts in zip(ax, times):
        fe, fs = _load_field(model_dir, "f_eq", ts), _load_field(model_dir, "f_sup", ts)
        z = np.arange(len(fe)) / 32
        a.axhline(0, color=ZERO, lw=0.5)
        a.plot(z, fe, color=ORANGE, lw=1.1, label=r"adiabatic $f_\mathrm{eq}$")
        a.plot(z, fs, color=BLUE, lw=1.1, label=r"superadiabatic $f_\mathrm{sup}$")
        a.plot(z, fe + fs, color=INK, lw=0.9, ls="--", label=r"internal $f_\mathrm{eq}+f_\mathrm{sup}$")
        a.set_title(f"$t={ts:.1f}\\tau$", fontsize=8)
        _grid(a)
        a.set_xlabel(LABEL_Z)
    ax[0].set_ylabel(r"force [$\epsilon/\sigma$]")
    rect = _top_legend(fig, ax[0], ncol=3)
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=rect)
    save(fig, out)
