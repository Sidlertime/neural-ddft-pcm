"""
Movie companion to plot_diffcoeff_comparison.py - same corrected structure
(same model_mode, same potential, varying only diffusion_coeff; combined
movies only ever vary model_mode within ONE potential, never across
potentials). Produces 3 movies per potential (hybrid, equilibrium,
combined), written into that potential's diffcoeff_comparison_{3a,3d}/
directory alongside the error plots.
"""
import os

from neural_ddft.utils.functions import plot_movie_multiple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same CELLS mapping as plot_diffcoeff_comparison.py (kept in sync manually -
# both scripts are small/stable enough that a shared import felt like
# unnecessary coupling for a one-off report-figures pair).
CELLS = {
    ("hybrid", "3a"): [
        ("results/2026-11-08_full_eval/diffcoeff0_3a_hybrid_01", "diffusion_coeff=0.0"),
        ("results/2026-11-06_full_eval/v7a_3a_hybrid_01", "diffusion_coeff=0.5"),
    ],
    ("hybrid", "3d"): [
        ("results/2026-11-08_full_eval/diffcoeff0_3d_hybrid_01", "diffusion_coeff=0.0 (blows up)"),
        ("results/2026-11-08_full_eval/diffcoeff025_3d_hybrid_01", "diffusion_coeff=0.25 (blows up)"),
        ("results/2026-11-07_full_eval/numerics_sweep_3d_dt_5e-5_full_hybrid_01", "diffusion_coeff=0.5"),
    ],
    ("equilibrium", "3a"): [
        ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_diffcoeff0_equilibrium_01", "diffusion_coeff=0.0"),
        ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_baseline_equilibrium_01", "diffusion_coeff=0.5"),
    ],
    ("equilibrium", "3d"): [
        ("results/2026-11-12_diffcoeff_stability/3d_equilibrium_diffcoeff0_equilibrium_01", "diffusion_coeff=0.0 (blows up)"),
        ("results/2026-11-12_diffcoeff_stability/3d_equilibrium_baseline_equilibrium_01", "diffusion_coeff=0.5 (survives)"),
    ],
}

MD_PATHS = {
    "3a": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
}


def movie_single_model(model_mode, potential, out_dir):
    ddft_configs = [(path, f"{model_mode}, {label}") for path, label in CELLS[(model_mode, potential)]]
    plot_movie_multiple(
        output_path=out_dir,
        ddft_configs=ddft_configs,
        md_configs=[(MD_PATHS[potential], "MD reference")],
    )


def movie_combined(potential, out_dir):
    ddft_configs = []
    for model_mode in ("hybrid", "equilibrium"):
        ddft_configs += [(path, f"{model_mode}, {label}") for path, label in CELLS[(model_mode, potential)]]
    plot_movie_multiple(
        output_path=out_dir,
        ddft_configs=ddft_configs,
        md_configs=[(MD_PATHS[potential], "MD reference")],
    )


def main():
    os.chdir(REPO_ROOT)

    for potential in ("3a", "3d"):
        base = f"results/report_figures/diffcoeff_comparison_{potential}"
        os.makedirs(base, exist_ok=True)
        movie_single_model("hybrid", potential, f"{base}/movie_hybrid")
        movie_single_model("equilibrium", potential, f"{base}/movie_equilibrium")
        movie_combined(potential, f"{base}/movie_combined")
        print(f"Done: {potential}")


if __name__ == "__main__":
    main()
