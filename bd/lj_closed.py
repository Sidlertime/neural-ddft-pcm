"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
from lammps import PyLammps
from mdext import log

from utils import plot, plot_movie, avg_results
import argparse
import os
from datetime import datetime

def main(args) -> None:

    # Some simulation parameters:
    T = 2
    P = None
    dz = 1/100
    collection_steps = args.collection_steps
    equilibration_steps = args.equilibration_steps
    dt = args.dt

    np.random.seed(args.pot_seed)

    current_time = datetime.now().strftime("%Y-%m-%d")
    results_path = f"./results/{current_time}_{args.run_id}_t{args.trials}_cs{args.collection_steps}_es{args.equilibration_steps}_g{args.gamma}_dt{args.dt}"
    os.makedirs(results_path, exist_ok=True)
    rnd_potential = mdext.potential.RandomGaussianSum()

    for i in range(args.trials):
        # Initialize and run simulation:
        md = mdext.md.MD(
            setup=setup,
            T=T,
            P=P,
            seed=i+1,
#            potential=mdext.potential.GaussianSum([0], [5], [1]),
            potential=mdext.potential.Gaussian(0, 5, 1),
            geometry_type=mdext.geometry.Planar,
            n_atom_types=1,
            potential_type=1,
            dr=dz,
            units="lj",
            timestep=dt,
            steps_per_thermo=10,
            thermo_per_cycle=1,
            gamma=args.gamma
        )

        os.makedirs(f"{results_path}/trials/{i}", exist_ok=True)
        md.run(equilibration_steps, "equilibration", f"{results_path}/trials/{i}/equilibration")

        md.reset_stats()
        md.reset_vext(rnd_potential)

        md.run(collection_steps, "collection", f"{results_path}/trials/{i}/collection")
        plot(f"{results_path}/trials/{i}")

    avg_results(results_path, args.trials, collection_steps)
    plot_movie(f"{results_path}/avg_{args.trials}")


def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct simulation box:
    L = np.array([10., 10., 10.])  # overall box dimensions
    lmp.region(
        f"sim_box block 0 {L[0]} 0 {L[1]} 0 {L[2]}"
        " units box"
    )
    lmp.create_box("1 sim_box")
    n_bulk = 0.6
    n_atoms = int(np.round(n_bulk * L.prod()))
    lmp.create_atoms(f"1 random {n_atoms} {seed} sim_box")
    lmp.mass("1 1.")

    # Interaction potential:
    lmp.pair_style("lj/cut 4")
    lmp.pair_coeff("1 1 1.0 1.0")

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="test")
    parser.add_argument("--pot_seed", type=int, default="28")
    parser.add_argument("--trials", type=int, default="1")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--collection_steps", type=int, default=100)
    parser.add_argument("--equilibration_steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1)
    main(parser.parse_args())