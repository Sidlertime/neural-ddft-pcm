"""
Create the report figures from the evaluation runs of scripts/reproduce/evaluate.py
(and the MD data). Missing evaluations are reported and their figures skipped.

Examples (run from neural-ddft-pcm/ddft):
    python scripts/reproduce/figures.py                       # all, into results/report/figures/
    python scripts/reproduce/figures.py --only rmse_vs_time_3a network_sketch
    python scripts/reproduce/figures.py --out_dir my_figures

The solver comparison (PhiFlow / pseudo-spectral / py-pde) is made with
scripts/evaluation/compare_runs.py (see the README).
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from neural_ddft.utils import experiments as X
from neural_ddft.utils import paths
from neural_ddft.utils import plotting as P


def network_sketch(out):
    """Static sketch of the hybrid model and its end-to-end training (Figure 1)."""
    P.apply_style()
    LB, LO, LG, EDGE = "#e3eefb", "#fde6dc", "#f2f2ef", "#9a9a94"

    def box(a, x, y, w, h, txt, fc, ec, fs=6.8):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
        a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=P.INK)

    def arr(a, x0, y0, x1, y1, c=P.INK, ls="-", rad=0.0, lw=0.8):
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(
            arrowstyle="-|>", lw=lw, color=c, ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={rad}"))

    fig = plt.figure(figsize=(6.9, 2.4))
    a = fig.add_axes([0, 0, 1, 1]); a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    box(a, 0.012, 0.36, 0.135, 0.50, "inputs\n$\\rho,\\ v$\n$\\partial_z\\rho,\\ \\partial_z v$\n$V_\\mathrm{ext}$\n"
        "$\\rho_\\mathrm{prev},\\ v_\\mathrm{prev}$\n$\\partial_t v,\\ \\gamma,\\ \\eta$", LG, EDGE)
    xs = [0.19, 0.27, 0.35, 0.43]
    for k, x in enumerate(xs):
        box(a, x, 0.48, 0.06, 0.30, "Conv1d\nk=5\n" + ("16 ch\nMish" if k < 3 else "1 ch"), LB, P.BLUE, fs=6.0)
    a.text(0.34, 0.42, "SuperadiabaticNet: circular padding, receptive field $0.53\\sigma$", ha="center", fontsize=6.2, color=P.BLUE)
    arr(a, 0.151, 0.63, 0.186, 0.63)
    for x in xs[:-1]:
        arr(a, x + 0.064, 0.63, x + 0.076, 0.63)
    box(a, 0.53, 0.55, 0.07, 0.16, "$f_\\mathrm{sup}$", LB, P.BLUE, fs=8)
    arr(a, 0.494, 0.63, 0.526, 0.63)
    box(a, 0.19, 0.07, 0.20, 0.20, "PCM functional $F_\\mathrm{exc}[\\rho]$\n(frozen)", LO, P.ORANGE)
    box(a, 0.53, 0.09, 0.07, 0.16, "$f_\\mathrm{eq}$", LO, P.ORANGE, fs=8)
    arr(a, 0.394, 0.17, 0.526, 0.17); a.text(0.46, 0.20, "$-\\partial_z\\,\\delta F/\\delta\\rho$", ha="center", fontsize=6.2)
    arr(a, 0.08, 0.355, 0.186, 0.17, rad=0.3); a.text(0.075, 0.22, "$\\rho$", fontsize=7)
    box(a, 0.66, 0.28, 0.14, 0.38, "PhiFlow step, Eq. (3)\n$dt=10^{-4}$\n$\\times100$ per MD frame\n$\\times W$ frames", LG, EDGE, fs=6.6)
    arr(a, 0.604, 0.63, 0.656, 0.56); arr(a, 0.604, 0.17, 0.656, 0.36)
    box(a, 0.845, 0.28, 0.145, 0.38, "loss vs. MD frames\n$w_\\rho\\,\\mathrm{MSE}(\\rho)$\n$+\\,w_v\\,\\mathrm{MSE}(v)$", LG, EDGE, fs=6.6)
    arr(a, 0.804, 0.47, 0.841, 0.47)
    arr(a, 0.917, 0.67, 0.46, 0.80, c=P.BLUE, ls="--", rad=0.18, lw=0.9)
    a.text(0.70, 0.93, "backpropagation through all solver steps (BPTT)", ha="center", fontsize=6.4, color=P.BLUE)
    P.save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=X.DEFAULT_MANIFEST)
    p.add_argument("--out_dir", default=os.path.join(paths.RESULTS_ROOT, "report", "figures"))
    p.add_argument("--only", nargs="*", help="figure names (default: all)")
    args = p.parse_args()

    m = X.load(args.manifest)
    d = m["defaults"]
    E = lambda model, target, **kw: X.Evaluation(model, target, kw.get("gamma", d["gamma"]), kw.get("dt", d["dt"]),
                                                  kw.get("t_range", d["t_range"]), kw.get("diffusion_coeff"))
    base3a, hyb3a = E(None, "3a"), E("density_weighted/3a", "3a")
    hybrid_label = "Hybrid (PCM + $f_\\mathrm{sup}$)"
    out = lambda name: os.path.join(args.out_dir, name)

    figures = {
        "network_sketch": ([], lambda: network_sketch(out("network_sketch"))),
        "potentials_overview": ([], lambda: P.potentials_overview(
            [(p_, paths.md_dir(p_)) for p_ in ["3a", "3d", "mg1", "s1"]], out("potentials_overview"))),
        "rmse_vs_time_3a": ([base3a, hyb3a], lambda: P.rmse_vs_time(
            [(base3a.out_dir(d), "PCM only", P.ORANGE), (hyb3a.out_dir(d), hybrid_label, P.BLUE)],
            paths.md_dir("3a"), out("rmse_vs_time_3a"))),
        "profile_snapshots_3a": ([base3a, hyb3a], lambda: P.profile_snapshots(
            [(base3a.out_dir(d), "PCM only", P.ORANGE), (hyb3a.out_dir(d), hybrid_label, P.BLUE)],
            paths.md_dir("3a"), out("profile_snapshots_3a"))),
        "flux_split_times_3a": ([base3a, hyb3a], lambda: P.flux_split_times(
            hyb3a.out_dir(d), base3a.out_dir(d), paths.md_dir("3a"), out("flux_split_times_3a"))),
        "flux_split_3a": ([base3a, hyb3a], lambda: P.flux_split_maps(
            hyb3a.out_dir(d), base3a.out_dir(d), paths.md_dir("3a"), out("flux_split_3a"))),
        "internal_force_split_3a": ([hyb3a], lambda: P.internal_force_split(hyb3a.out_dir(d), out("internal_force_split_3a"))),
        "artificial_diffusion": ([E(None, "3a"), E(None, "3a", diffusion_coeff=0.0), E(None, "3d"), E(None, "3d", diffusion_coeff=0.0)],
            lambda: P.density_rmse_panels([
                (f"{t} (PCM only)", paths.md_dir(t), [
                    (E(None, t).out_dir(d), "artificial diffusion 0.5 (default)", P.BLUE),
                    (E(None, t, diffusion_coeff=0.0).out_dir(d), "no artificial diffusion", P.ORANGE)])
                for t in ["3a", "3d"]], out("artificial_diffusion"))),
    }
    for name, (needs, make) in figures.items():
        if args.only and name not in args.only:
            continue
        missing = [e.name(d) for e in needs if not X.is_done(e, m)]
        if missing:
            print(f"skip {name}: missing evaluation(s) {', '.join(missing)} (run scripts/reproduce/evaluate.py)")
            continue
        make()


if __name__ == "__main__":
    main()
