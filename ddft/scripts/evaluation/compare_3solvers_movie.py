"""
3-solver comparison movies: PhiFlow / Goddard (pseudo-spectral) / py-pde,
each solver's equilibrium (base model only) and trained-hybrid-mode output
overlaid against the true MD trajectory, via plot_movie_multiple (already
solver-agnostic - reads rho*.npy/v*.npy by naming convention only).

Three scenarios (kept as separate movies, not combined):
  (a) 3a, equilibrium mode, all 3 solvers
  (b) 3a, one trained hybrid checkpoint, all 3 solvers
  (c) 3d, all 3 solvers (PhiFlow/Goddard expected to blow up; py-pde is a
      first-time unknown at this potential)

eta=0.6944 (established production value) used consistently across all 3
solvers, not the newly-measured 0.8053, per the plan's explicit choice.
"""
import os

from neural_ddft.utils.functions import plot_movie_multiple

# Sibling checkout of the pseudo-spectral solver (github.com/Sidlertime/goddard-ddft);
# override with the GODDARD_ROOT environment variable.
GODDARD_ROOT = os.environ.get("GODDARD_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "goddard")))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("results/report_figures", exist_ok=True)

    # (a) 3a, equilibrium mode, all 3 solvers
    # NOTE: the PhiFlow source below was previously
    # "results/2026-11-08_goddard_compare_phiflow_eq_equilibrium_01", a
    # stale run on the GAUSSIAN potential, reused by mistake - fixed to the
    # correct 3a equilibrium baseline run.
    plot_movie_multiple(
        output_path="results/report_figures/3solver_3a_equilibrium",
        ddft_configs=[
            ("results/2026-11-12_diffcoeff_stability/3a_equilibrium_baseline_equilibrium_01", "PhiFlow"),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-08_ddft_neural_3a_compare_01"), "Goddard (pseudo-spectral)"),
            ("results/2026-11-12_pypde_smoketest_3a_01", "py-pde"),
        ],
        md_configs=[("../data/lj_data/ref_3a_langevin_01/avg_2000/", "MD reference")],
    )

    # (b) 3a, trained hybrid checkpoint (vext combo), all 3 solvers
    # NOTE: "3a_hybrid_vext_full_hybrid_01" (previous source) silently used
    # the GAUSSIAN potential too - phiflow_eval.py's PARAM_SCHEME key is
    # "md_path" (singular) but training configs use "md_paths" (plural,
    # list), so passing --config alone without an explicit --md_path CLI
    # override silently falls back to the gaussian default, no error raised.
    # Fixed by always passing --md_path explicitly (see "_fixed" rerun).
    plot_movie_multiple(
        output_path="results/report_figures/3solver_3a_hybrid",
        ddft_configs=[
            ("results/2026-11-12_3solver_compare/3a_hybrid_vext_full_fixed_hybrid_01", "PhiFlow"),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-12_goddard_hybrid_smoketest_3a_03"), "Goddard (pseudo-spectral)"),
            ("results/2026-11-12_pypde_hybrid_smoketest_3a_01", "py-pde"),
        ],
        md_configs=[("../data/lj_data/ref_3a_langevin_01/avg_2000/", "MD reference")],
    )

    # (c) 3d, same zero-shot hybrid checkpoint, all 3 solvers (none blow up -
    # the zero-shot correction stabilizes all 3, in contrast to py-pde's own
    # equilibrium-mode 3d run, which goes to NaN)
    plot_movie_multiple(
        output_path="results/report_figures/3solver_3d",
        ddft_configs=[
            ("results/2026-11-12_3solver_compare/3d_hybrid_zeroshot_vext_hybrid_02", "PhiFlow (zero-shot)"),
            (os.path.join(GODDARD_ROOT, "results", "2026-11-12_goddard_3d_zeroshot_vext_01"), "Goddard (zero-shot)"),
            ("results/2026-11-12_pypde_3d_zeroshot_vext_01", "py-pde (zero-shot)"),
        ],
        md_configs=[("../data/lj_data/ref_3d_langevin_01/avg_2000/", "MD reference")],
    )


if __name__ == "__main__":
    main()
