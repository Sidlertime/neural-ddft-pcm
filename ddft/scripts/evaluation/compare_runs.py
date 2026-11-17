"""
Compare any number of evaluation runs against the MD reference: density and
velocity RMSE over time, profile snapshots, and optionally a movie.

A run is a folder with rho*.npy / v*.npy per saved frame, as written by
scripts/evaluation/phiflow_eval.py, the pseudo-spectral solver
(goddard-ddft: scripts/ddft_neural.py) or legacy/scripts/evaluation/pypde_eval.py.
Frames are matched to MD by time.

Examples (run from neural-ddft-pcm/ddft):
    # PCM-only model with the three solvers (numerical validation)
    python scripts/evaluation/compare_runs.py --md 3a --out results/comparisons/3a_solvers \\
        --runs results/report/eval/baseline/3a:PhiFlow \\
               $GODDARD_ROOT/results/<date>_psm_3a_01:pseudo-spectral \\
               results/<date>_pypde_3a_01:py-pde

    # two models on the same potential, with a movie
    python scripts/evaluation/compare_runs.py --md mg1 --movie --out results/comparisons/mg1 \\
        --runs results/report/eval/baseline/mg1:"PCM only" results/report/eval/density_weighted/mg1/mg1:hybrid
"""
import argparse
import os

from neural_ddft.utils import paths
from neural_ddft.utils import plotting as P


def parse_run(spec):
    """'path[:label]' -> (path, label); the label defaults to the folder name."""
    path, _, label = spec.partition(":") if not os.path.exists(spec) else (spec, "", "")
    return paths.resolve(path), label or os.path.basename(os.path.normpath(path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders, each as path[:label]")
    p.add_argument("--md", required=True, help="potential name (e.g. 3a) or path to an MD folder with collection.*.h5")
    p.add_argument("--gamma", type=float, default=paths.DEFAULT_GAMMA, help="friction of the MD data if --md is a name")
    p.add_argument("--out", required=True, help="output prefix, e.g. results/comparisons/3a_solvers")
    p.add_argument("--snapshots", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0], help="times of the profile snapshots")
    p.add_argument("--movie", action="store_true", help="also render an animated comparison (mp4)")
    args = p.parse_args()

    md = paths.resolve(args.md) if os.path.isdir(paths.resolve(args.md)) else paths.md_dir(args.md, args.gamma)
    runs = [parse_run(r) for r in args.runs]
    missing = [r for r, _ in runs if not os.path.isdir(r)]
    if missing:
        p.error("run folder(s) not found: " + ", ".join(missing))

    P.rmse_vs_time(runs, md, args.out + "_rmse")
    P.profile_snapshots(runs, md, args.out + "_snapshots", times=args.snapshots)
    if args.movie:
        from neural_ddft.utils.functions import plot_movie_multiple
        movie_dir = args.out + "_movie"
        os.makedirs(movie_dir, exist_ok=True)
        plot_movie_multiple(movie_dir, ddft_configs=runs, md_configs=[(md, "MD")])
        print(f"saved movie to {movie_dir}/")


if __name__ == "__main__":
    main()
