# Learned Free-Energy Functionals from Pair-Correlation Matching for Dynamical Density Functional Theory
*(Extended for Underdamped Dynamical Simulations)*

[PR-E article](https://doi.org/10.1103/22fd-ykkb) | [Preprint](https://arxiv.org/abs/2505.09543) | [Data](10.5281/zenodo.17116150)

![Breakthrough simulation](breakthrough_simulation.gif)

DDFT and BD simulation code to reproduce the results in the PR-E article, now extended to support **underdamped Langevin dynamics** using a differentiable PhiFlow PDE solver. Pre-computed simulation data is available on [Zenodo](10.5281/zenodo.17116150).

## Repository Architecture

This project has been updated to use a modern Python `src` layout to prevent import collisions and enable clean namespace imports across scripts.

*   `src/neural_ddft/`: The core library namespace.
    *   `models/`: PyTorch architectures (ZimmermannNet, SuperadiabaticNet, NeuralFreeEnergy).
    *   `physics/`: The differentiable PhiFlow PDE integrator and the unified Force Factory.
    *   `data/`: Sliding-window dataset loaders for HDF5 MD trajectories.
    *   `utils/`: Configuration management and plotting tools.
*   `scripts/`: Executable entry points for evaluation and training.
*   `bd/`: Scripts for simulating the reference Langevin and Brownian dynamics data.
*   `ddft/`: Legacy scripts for the original overdamped DDFT solvers.

## Setup Environment

Create the environment and install the package in editable mode using the modern `pyproject.toml` setup. This allows you to import the library (`from neural_ddft import ...`) from anywhere on your machine.

```bash
mamba env create -f environment.yml # or conda
mamba activate neural-ddft
pip install -e .
```

## Data Generation (Brownian & Langevin Dynamics)

To generate the reference molecular dynamics data:

```bash
# 1. Original Overdamped BD (Closed & Open)
cd bd
python lj_closed.py --collection_steps=400 --equilibration_steps=200 --trials=100 --gamma=8 --dt=0.001
python lj_open.py --collection_steps=1500 --equilibration_steps=200 --trials=100 --gamma=10 --dt=0.001

# 2. Underdamped LD (for inertial DDFT)
# Must have mdext-ddft installed (e.g. on same level as neural-ddft-pcm)
# Note: Requires higher trials (e.g. 500) to reduce noise in the velocity fields
cd mdext_-_ddft/examples/lj
python lj.py --collection_steps=400 --equilibration_steps=200 --trials=500 --gamma=0.1 --dt=0.0001
```

5000 trials were used for the closed system and 500 trials were used for the open system in the original publication. The external potential is set inside the script. Note that steps here refer to the number of cycles, and each cycle consists of 10 and 100 steps for the closed and open system, respectively.

## Training Methodology: End-to-End Unrolled PDE Training

For underdamped inertial systems, we utilize End-to-End Unrolled PDE Training rather than relying on direct supervised learning of the internal forces.

By keeping the PyTorch computational graph alive during the PhiFlow integration, the network is trained by evaluating the Mean Squared Error (MSE) directly on the future density $\rho$ and velocity $v$ states. To combat autoregressive exposure bias (compounding errors over long rollouts), the system uses a Long-Horizon curriculum that progressively increases the PDE unroll window from 2 to 25 timesteps.

To execute a training curriculum:

```bash
python scripts/training/train_unified.py --config configs/zimmermann_long_horizon.json
```

## Evaluating DDFT Simulations

### Underdamped Regime (New PhiFlow Solver)

To evaluate trained models (Zimmermann or Hybrid) using the new fully coupled continuity and momentum equations:

```bash
python scripts/evaluation/run_phiflow_eval.py --config configs/eval_zimmermann.json
```

### Overdamped Regime (Legacy Solvers)

To reproduce the original results from the PR-E article:

```bash
cd ddft
python ddft_neural_closed.py --md_path=../data/g1-closed-dz100/avg_5000
python ddft_neural_open.py --md_path=../data/irmof1-open-dz32/avg_500
```

Use `--type=c1` to use the c1 model and the `ddft_fmt.py` script to use the FMT model. Note that the closed script uses `dz=1/100` and `dt=0.0001`, and the open script uses `dz=1/32` and `dt=0.001`.

## Numerical Methods & Physics Integrator   

The new underdamped PhiFlowDDFT integrator fundamentally differs from standard finite-difference solvers (like `py-pde`) to guarantee stability in highly inertial regimes ($\gamma = 0.1$):

1. **Staggered Grids:** Density $\rho$ is stored at cell centers, while velocity $v$ is stored on the cell faces. This tightly couples the momentum and density fields, preventing numerical pressure checkerboarding.
2. **Semi-Implicit Euler:** The solver updates the density first, then uses the future density $\rho(t+\Delta t)$ to calculate the pressure and excess forces that drive the velocity update, strictly conserving momentum.
3. **Semi-Lagrangian Advection:** Rather than using strict Eulerian gradients to advect velocity (which are constrained by the CFL condition), the solver traces fluid packets backwards in time to their previous locations, making advection unconditionally stable.
4. **Artificial Diffusion (Upwind Stabilization):** To prevent high-frequency numerical dispersion (Gibbs phenomenon) at sharp density interfaces, a dynamic artificial diffusion term is injected into the continuity equation:
   $$Flux_{diffusive} = - \frac{1}{2} \Delta x \vert{}v\vert{} \nabla \rho$$
   This acts as a targeted numerical shock absorber that scales with grid resolution, smoothing the fluid only where steep gradients and high velocities threaten to break the solver.

## Acknowledgements

[mdext](https://github.com/shankar1729/mdext)<br>
[py-pde](https://github.com/zwicker-group/py-pde)<br>
[neural-free-energy-1d](https://github.com/jacobusdijkman/neural-free-energy-1d/tree/main)<br>
[PyDFTlj](https://github.com/elvissoares/PyDFTlj)
