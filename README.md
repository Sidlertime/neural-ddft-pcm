# Neural DDFT for Underdamped Dynamics: PCM Functional + Learned Superadiabatic Force

![Breakthrough simulation](breakthrough_simulation.gif)

This repository extends the code of
[Ram et al. (2025)](https://doi.org/10.1103/22fd-ykkb), *Learned Free-Energy Functionals from
Pair-Correlation Matching for Dynamical Density Functional Theory*, from **overdamped** to
**underdamped** (inertial) dynamical density functional theory (DDFT).

The underdamped DDFT of Archer (2009) evolves the one-body density ρ(z,t) and velocity
v(z,t) of a fluid in an external potential. The internal force has two parts:
- the **adiabatic** force f_eq = −∂_z δF_exc/δρ, from the pair-correlation-matching (PCM)
  neural free-energy functional of Dijkman et al. (2025), kept frozen;
- a **superadiabatic** correction f_sup, learned by a small convolutional network.

The network is trained end-to-end through a differentiable PDE solver
([PhiFlow](https://github.com/tum-pbs/PhiFlow)) on density and velocity profiles from Langevin
molecular dynamics (MD), without any force labels. On the reference potentials it reduces
the velocity error of the PCM-only model by 20–60% and, with a density-weighted loss, the
density error on most potentials.

This was done as an interdisciplinary project (IDP) at the Technical University of Munich.
The repository contains:
- the solver, models and training/evaluation pipeline;
- all trained models of the report;
- scripts that reproduce every table and figure of the report.

**Related repositories** (clone all three next to each other, see [Installation](#installation)):

| Repository | Purpose |
|---|---|
| [mdext-ddft](https://github.com/Sidlertime/mdext-ddft) | Generates the Langevin MD reference data (density + velocity profiles), the external potentials and the viscosity measurement |
| [goddard-ddft](https://github.com/Sidlertime/goddard-ddft) | Pseudo-spectral underdamped DDFT solver, used for numerical validation |
| neural-ddft-pcm (this) | PhiFlow solver, hybrid model, training, evaluation, report reproduction |

## Contents

```
neural-ddft-pcm/
├── ddft/                       working directory of all commands below
│   ├── src/neural_ddft/        Python package
│   │   ├── physics/            PhiFlow underdamped DDFT solver, force computation (f_eq, f_sup)
│   │   ├── models/             PCM functional, SuperadiabaticNet, alternative outputs
│   │   ├── data/               MD trajectory datasets for training
│   │   └── utils/              config, paths, metrics, plotting, experiment manifest
│   ├── scripts/
│   │   ├── training/           train_phiflow.py (end-to-end training)
│   │   ├── evaluation/         phiflow_eval.py (simulate a model), compare_runs.py (plots)
│   │   ├── reproduce/          train / evaluate / tables / figures of the report
│   │   └── tools/              diagnostics (continuity check, profile comparison, ...)
│   ├── configs/report/         training configs of all report models
│   ├── experiments/report.yaml which model, data and baseline belong to which report table
│   ├── models/                 PCM functionals (c1/c2, Ram et al.) and report/ (trained models)
│   ├── tests/                  pytest suite
│   └── legacy/                 original overdamped solvers of Ram et al. (py-pde)
├── bd/                         Brownian dynamics data generation of Ram et al.
├── scripts/                    figure scripts of the original PR-E article
└── data/                       MD data (not in git, see "Data")
```

## Installation

**Platforms:** Linux, macOS and Windows. The environment solves on all three; it was installed
and tested on Linux. MD data generation (mdext-ddft) needs LAMMPS, which is available for Linux
and macOS only (on Windows use WSL2).

```bash
mkdir workspace && cd workspace
git clone https://github.com/Sidlertime/neural-ddft-pcm.git
git clone https://github.com/Sidlertime/goddard-ddft.git goddard   # optional: numerical validation
git clone https://github.com/Sidlertime/mdext-ddft.git             # optional: generate MD data

cd neural-ddft-pcm
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate neural-ddft               # installs the neural_ddft package in editable mode
```

- **GPU:** the environment installs the CPU build of PyTorch. Evaluation runs fine on a CPU.
  Training is much faster on a GPU. Install a CUDA build of PyTorch into the environment, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu126` (see
  [pytorch.org](https://pytorch.org/get-started/locally/)).
- **Pinned versions:** PhiFlow 3.4.0 with PhiML 1.15.1. PhiML 1.16 breaks the semi-Lagrangian
  advection step of the solver.
- **Locations** are set in `ddft/src/neural_ddft/utils/paths.py`. Each can be overridden with an
  environment variable: `DATA_ROOT` (MD data, default `data/lj_data`), `RESULTS_ROOT` (default
  `ddft/results`), `GODDARD_ROOT` (default `../goddard`), `MDEXT_ROOT` (default
  `../mdext-ddft`).

### Tests

```bash
cd ddft
pytest tests          # about 1 min on a CPU
```

The tests cover the grid operators, the PDE step and rollout, the force computation, the
datasets and the alternative output models. They include a regression test that the training
unroll (100 solver substeps per MD frame) reproduces the evaluation solver step by step. Tests
that need MD data are skipped if it is missing.

## Data

| What | Where it comes from | Where it goes |
|---|---|---|
| Langevin MD reference data (density + velocity, 400 frames of 0.01τ) | generated with [mdext-ddft](https://github.com/Sidlertime/mdext-ddft): `examples/lj/run_collection.sh` | `data/lj_data/ref_<potential>[_gamma<γ>]_langevin_01/avg_<trials>/` |
| PCM free-energy functionals `c1`, `c2` | Ram et al. (2025), included | `ddft/models/c*_dz*.ckpt` |
| Trained correction models of the report | included | `ddft/models/report/` |
| Brownian dynamics data of Ram et al. (overdamped, legacy only) | [Zenodo](https://doi.org/10.5281/zenodo.17116150) (CC BY 4.0) | `data/<name>-closed-dz100/` (unzipped) |

The MD data is not part of the repository. To create it, follow the
[mdext-ddft README](https://github.com/Sidlertime/mdext-ddft#reproducing-the-reference-data).
In short (about 2.4 h per potential with 500 trials on one core; run potentials in parallel):

```bash
cd ../mdext-ddft/examples/lj
python potentials.py                       # mg1, mg2, s1, s2, s3, gausin1
python potentials.py --reference           # 3a-3d from the Zenodo data of Ram et al.
./run_collection.sh gaussian mg1 mg2 s1 s2 s3 gausin1
TRIALS=2000 ./run_collection.sh 3a 3b 3c 3d
TDAMP=20 TAG=ref_gamma005 ./run_collection.sh 3a gaussian      # gamma = 0.05
TDAMP=2  TAG=ref_gamma05  ./run_collection.sh 3a gaussian      # gamma = 0.5
```

Then link every `results/<date>_<tag>_<potential>_langevin_01/avg_<trials>` folder to
`neural-ddft-pcm/data/lj_data/ref_<potential>[_gamma005|_gamma05]_langevin_01/avg_<trials>`.

## Reproducing the report

All commands run in `ddft/`. The manifest [`experiments/report.yaml`](ddft/experiments/report.yaml)
lists:
- every model, with its training config and seed;
- every table, with each row's model, target potential, friction and time step.

Every reported value is the relative change of the trajectory RMSE vs. the PCM-only model on
the same data, Δ = (RMSE_model − RMSE_PCM) / RMSE_PCM; negative is better.

### Quick: from the included models (CPU, no training)

```bash
python scripts/reproduce/evaluate.py --dry_run    # lists the 69 evaluations
python scripts/reproduce/evaluate.py --jobs 8     # runs them, resumable
python scripts/reproduce/tables.py                # all tables (Markdown; --latex for LaTeX)
python scripts/reproduce/figures.py               # all figures -> results/report/figures/
```

- **Runtime:** one evaluation simulates 4τ (40,000 solver steps) and takes about 20–30 min on
  one CPU core, so 69 evaluations take about 3–4 h with `--jobs 8`.
- **Subsets:** `--tables self inputs` runs only the evaluations of those tables. `figures.py`
  only needs the 3a evaluations (`--tables self`) and the PCM-only runs listed under
  `figures_extra`.
- **Outputs:** `results/report/eval/<model>/<target>/` and `results/report/eval/baseline/<target>/`.

### Full: from MD data to tables

```bash
python scripts/reproduce/train.py --jobs 3          # 24 models, about 45-60 min each on a GPU
python scripts/reproduce/evaluate.py --models trained --jobs 8
python scripts/reproduce/tables.py && python scripts/reproduce/figures.py
```

`train.py --only 'density_weighted/*'` trains a subset. Training is not bit-wise
deterministic on a GPU, so retrained models give slightly different numbers.

### Report sections and where they come from

| Report | How to reproduce |
|---|---|
| Tables 1–6, Sec. 6.3 and 6.6, alternative outputs | `tables.py` (manifest tables `self`, `loss_balance`, `zero_shot`, `friction`, `inputs`, `multi_potential`, `alternative_outputs`) |
| Figures 1–5 and appendix figures | `figures.py` |
| Hyperparameters, potential table | `configs/report/*.yaml`, mdext-ddft `examples/lj/potentials.py` |
| Numerical validation: PhiFlow vs. pseudo-spectral vs. py-pde | see below |
| Continuity check (integrator error with MD velocity) | `python scripts/tools/continuity_diagnostic.py --help` |
| Viscosity η = 0.81 ± 0.07 | mdext-ddft README, "Viscosity (Green-Kubo)" |

**Numerical validation (3-solver comparison):** run the PCM-only model with the pseudo-spectral
solver (see the [goddard-ddft README](https://github.com/Sidlertime/goddard-ddft#reproducing-the-results))
and with py-pde, then compare with the PhiFlow baseline:

```bash
python legacy/scripts/evaluation/pypde_eval.py --run_id pypde_3a --type c2 --gamma 0.1 --eta 0.6944 \
    --dt 1e-4 --md_path ../data/lj_data/ref_3a_langevin_01/avg_2000
python scripts/evaluation/compare_runs.py --md 3a --out results/comparisons/3a_solvers \
    --runs results/report/eval/baseline/3a:PhiFlow \
           ../../goddard/results/<date>_psm_3a_01:pseudo-spectral \
           results/<date>_pypde_3a_01:py-pde
```

### Differences to the submitted report

- **Frame pairing.** The evaluations start from MD frame 0 and save at t = 0.01, 0.02, … τ.
  The report compared the i-th saved prediction with MD frame i, i.e. one frame (0.01τ) too
  early. The metrics now pair frames by time.
  - This shifts all relative changes by 0.1–1.5 percentage points; no conclusion changes.
  - `tables.py --report_pairing` reproduces the report values exactly.
- **Table 6 (4-potential model).** Its evaluations were stopped at t ≈ 3.66τ by the
  30-minute wall-clock limit of `phiflow_eval.py`, not by a divergence. The reproduction
  scripts use a 4 h limit and evaluate the full 4τ.
- **Alternative outputs.** The smoke tests trained for 10 epochs (the report states 20).

## Using the code for new experiments

- **More trials, other potentials or other frictions:** generate the data with mdext-ddft:
  - more trials: `TRIALS=5000 ./run_collection.sh 3a`;
  - a new potential: add a formula to `potentials.py`;
  - a new friction: `TDAMP=1/γ`.
  
  Link the result to `data/lj_data/ref_<name>[_gamma<γ>]_langevin_01/`, e.g. `_gamma02` for
  γ = 0.2. `paths.md_dir(name, gamma)` then finds it.
- **Train a model:** copy a config from `configs/report/density_weighted/`, set `md_paths`, `run_id`,
  `gamma` and `loss_weight_rho`. A good starting point for the density weight is
  w_ρ = 2 · L_v / L_ρ, from the losses of the first epoch; it may need to be doubled.

  ```bash
  python scripts/training/train_phiflow.py --config my_config.yaml --out_dir results/my_model
  ```

- **Evaluate any model on any data:** use the training config for the input channels and
  physics:

  ```bash
  python scripts/evaluation/phiflow_eval.py --config my_config.yaml --model_mode hybrid \
      --sup_model_path results/my_model/checkpoints/hybrid_epoch_080.ckpt \
      --md_path ../data/lj_data/ref_s2_langevin_01/avg_500 --eta 0.6944 --out_dir results/my_eval
  python scripts/evaluation/phiflow_eval.py --model_mode equilibrium --eta 0.6944 \
      --md_path ../data/lj_data/ref_s2_langevin_01/avg_500 --out_dir results/pcm_s2
  python scripts/evaluation/compare_runs.py --md s2 --runs results/pcm_s2:"PCM only" results/my_eval:hybrid \
      --out results/comparisons/s2 --movie
  ```

  Δ values: `neural_ddft.utils.metrics.relative_change(model_dir, baseline_dir, md_dir)`.
- **Limits of the explicit solver:** the viscosity term limits the time step to
  dt ≲ dz² m ρ_min / (2η). That is about 1.3·10⁻⁴ for 3a, but only about 2·10⁻⁵ for potentials
  with nearly empty regions (3b, 3d); use `--dt` accordingly.

## Overdamped DDFT (original PR-E article)

The original overdamped solvers of Ram et al. are kept in `ddft/legacy/` (they use py-pde).
Download and unzip the BD data from [Zenodo](https://doi.org/10.5281/zenodo.17116150) into
`data/`, then from `ddft/`:

```bash
python legacy/scripts/evaluation/ddft_neural_closed.py --md_path ../data/g1-closed-dz100/avg_5000
python legacy/scripts/evaluation/ddft_neural_open.py --md_path ../data/irmof1-open-dz32/avg_500
```

- **Other functionals:** `--type c1` uses the c1 model, and `ddft_fmt.py` the FMT functional.
- **Grids:** the closed script uses dz = 1/100 and dt = 10⁻⁴, the open script dz = 1/32 and
  dt = 10⁻³.
- **BD data generation:** `bd/` (`lj_closed.py`, `lj_open.py`, needs LAMMPS).
- **Figures of the article:** `scripts/plot_fig*.py`.

## Numerical method

The PhiFlow solver (`ddft/src/neural_ddft/physics/phiflow_pde.py`) integrates the planar
underdamped DDFT on a periodic box of 10σ with 320 grid points (dz = σ/32) and dt = 10⁻⁴τ:

1. **Staggered grid:** ρ at cell centres, v at cell faces.
2. **Semi-implicit Euler:** the continuity equation is updated first. Pressure, viscosity and
   internal force in the momentum update then use the new density.
3. **Semi-Lagrangian advection** of v.
4. **Artificial diffusion** in the continuity equation, flux −½ dz |v| ∂_z ρ, against numerical
   oscillations at steep density gradients. A density floor of 10⁻⁶ is also applied.

Training unrolls this solver for 100 substeps per MD frame (frames are 0.01τ apart) over
windows of 1–10 frames and backpropagates through all steps.

## Citation and attribution

- Original method and code: K. Ram, J. Dijkman, R. van Roij, J.-W. van de Meent, B. Ensing,
  M. Welling, D. Cremers, *Learned Free-Energy Functionals from Pair-Correlation Matching for
  Dynamical Density Functional Theory*, Phys. Rev. E (2025),
  [doi:10.1103/22fd-ykkb](https://doi.org/10.1103/22fd-ykkb),
  [arXiv:2505.09543](https://arxiv.org/abs/2505.09543).
- PCM functional: J. Dijkman et al., *Learning Neural Free-Energy Functionals with
  Pair-Correlation Matching*, Phys. Rev. Lett. 134, 056103 (2025).
- Underdamped DDFT: A. J. Archer, J. Chem. Phys. 130, 014509 (2009).
- Underdamped extension: J.-P. Sindermann, IDP report, Technical University of Munich (2026).

See [NOTICE.md](NOTICE.md) for licensing and attribution. Parts of the code and documentation
were written with the help of AI assistants (Anthropic Claude, Google Gemini).

## Acknowledgements

[mdext](https://github.com/shankar1729/mdext) ·
[PhiFlow](https://github.com/tum-pbs/PhiFlow) ·
[py-pde](https://github.com/zwicker-group/py-pde) ·
[neural-free-energy-1d](https://github.com/jacobusdijkman/neural-free-energy-1d) ·
[PyDFTlj](https://github.com/elvissoares/PyDFTlj)
