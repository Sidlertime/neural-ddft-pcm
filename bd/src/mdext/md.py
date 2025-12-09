from lammps import lammps, PyLammps
import mdext
from mdext import MPI
from .geometry import GeometryType, Spherical, Cylindrical, Planar
from .potential import Potential, ForceCallback
import numpy as np
import os
import time
import logging
import sys
import h5py
from typing import Callable, Optional


log: logging.Logger = logging.getLogger("mdext")  #: Log for the mdext module
log.setLevel(logging.WARNING if MPI.COMM_WORLD.rank else logging.INFO)
log.addHandler(logging.StreamHandler(sys.stderr))  # because PyLAMMPS captures stdout

thermo_callback = None  #: imported into __main__ by lammps python command

unit_names = {
    'real': {
        'mass': 'amu',
        'distance': 'A',
        'time': 'fs',
        'energy': 'kcal/mol',
        'temperature': 'K',
        'pressure': 'atm',
    },
    'metal': {
        'mass': 'amu',
        'distance': 'A',
        'time': 'ps',
        'energy': 'eV',
        'temperature': 'K',
        'pressure': 'bar',
    },
    'electron': {
        'mass': 'amu',
        'distance': 'a0',
        'time': 'fs',
        'energy': 'Eh',
        'temperature': 'K',
        'pressure': 'Pa',
    },
    'lj': {
        'mass': 'm_lj',
        'distance': 'sigma',
        'time': 'sigma sqrt(m_lj / epsilon)',
        'energy': 'epsilon',
        'temperature': 'epsilon',
        'pressure': 'epsilon / sigma^3',
    },
}  #: unit names (for logging)

 
class MD:
        
    def __init__(
        self,
        *,
        setup: Callable[[PyLammps, int], None],
        T: float,
        P: Optional[float],
        seed: int,
        potential: Potential,
        geometry_type: GeometryType,
        n_atom_types: int,
        potential_type: int,
        pe_collect_interval: int = 0,
        units: str = "real",
        dimension: int = 3,
        timestep: float = 2.0,
        steps_per_thermo: int = 50,
        thermo_per_cycle: int = 100,
        dr: float = 0.05,
        gamma: float = 1.0,
        enable_gcmc: bool = False,
        gcmc_region1: str = "atom_box1",
        gcmc_region2: str = "atom_box2",
        gcmc_mu1: float = 0.5,
        gcmc_mu2: float = -100.0,
        gcmc_temp: Optional[float] = None,
        prune_regions: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        setup
            Callable with signature `setup(lmp, seed)` that creates initial atomic
            configuration and sets up the interaction potential / force fields.
            This could load a LAMMPS data file, or use LAMMPS box and random atom
            creation commands. If starting with a random configuration, this should
            also ideally invoke minimize to ensure a reasonable starting point.
            (Don't need to equilibrate here; that can be done using `MD.run` later.)
        T
            Temperature in LAMMPS temperature units.
        P
            Pressure in LAMMPS pressure units for NPT, or `None` for NVT.
        seed
            Random seed for velocity generation and passed to `setup`.
        potential
            Function or callable calculating potential and derivative.
            This function should return E and dE/d(r_sq) given r_sq,
            the square of the 1D coordinate specified by `geometry_type`.
        geometry_type
            One of the geometry classes from `mdext.geometry` which specifies
            which reduced 1D coordinate to apply potentials and collect densities
            as a function of.
        n_atom_types
            Number of atom types in the LAMMPS simulation.
        potential_type
            1-based LAMMPS atom type that the external potential should be applied to.
            If zero, apply same potential to all the atoms.
        pe_collect_interval
            If non-zero, collect PE and minimum particle distance for cavitation
            analysis. Typically used only with repulsive spherical potentials.
        units
            Supported LAMMPS units system (see `unit_names`).
        dimension
            Set dimensionality of the system, by default 3.
            The LAMMPS dimension is always set to 3, with the other dimensionalities
            enforced by fixes. The initial configuration must have all atoms with
            z = 0 in 2D, and x = y = 0 in 1D. Only Cylindrical geometry is supported
            for 2D, and only Planar geometry is supported for 1D.
        timestep
            MD timestep in LAMMPS time units.
        steps_per_thermo
            Number of time steps between each thermo call.
        thermo_per_cycle
            Number of thermo calls per reporting cycle.
            Averaged thermo output is reported at this interval,
            and if requested, densities are updated to file at this interval.
        dr
            Spatial resolution for density collection in LAMMPS distance units.
            Note that densities are collected on a 1D planar, cylindrical or spherical
            grid based on whether `potential` is planar, cylindrical or spherical.
        gamma
            Damping coefficient for Brownian dynamics (in LAMMPS units).
        """
    
        # Set up LAMMPS instance:
        self.is_head = (MPI.COMM_WORLD.rank == 0)
        os.environ["OMP_NUM_THREADS"] = "1"  # run single-threaded
        lps = lammps()
        lmp = PyLammps(ptr=lps)
        self.lps = lps  #: Raw LAMMPS interface
        self.lmp = lmp  #: PyLAMMPS interface
        self.t_start = time.time()
    
        # Global settings:
        assert units in unit_names
        self.units = units
        self.unit_names = unit_names[units]
        lmp.units(units)
        
        # Set up dimensionality and boundary conditions:
        self.dimension = dimension
        lmp.dimension("3")  # LAMMPS always in 3D; dimensionality emulated by fixes.
        
        self.enable_gcmc = enable_gcmc
        self.prune_regions = prune_regions

        # Allow non-periodic z for lj_open-style setups using wall/reflect
        if self.enable_gcmc:
            lmp.boundary("p p f")
        else:
            lmp.boundary("p p p")
        
        # Set up initial atomic configuration and interaction potential:
        setup(lmp, seed)
    
        # Prepare for dynamics:
        lmp.reset_timestep(0)
        lmp.timestep(timestep)  # in time units consistent with 'units'
        self.T = T
        self.P = P
        self.gamma = gamma
     
        # Apply Brownian dynamics fix:
        #lmp.fix(f"Ensemble all brownian {T} {seed} gamma_t {self.gamma}")
    
        # OR Apply Langevin dynamics:
        # 1. Thermostat
        damp = 1.0 / self.gamma     # damp = mass / friction
        lmp.fix(f"therm all langevin {T} {T} {damp} {seed}")

        # 2. Integrator (NVE)
        lmp.fix("Ensemble all nve")

        # Initial velocities set to zero (optional):
        lmp.velocity("all set 0.0 0.0 0.0")
    
        # Setup thermo callback
        lmp.thermo(steps_per_thermo)
        mdext.md.thermo_callback = self
        lmp.python(
            "thermo_callback input 1 SELF return v_thermo_callback format pf here"
            " 'from mdext.md import thermo_callback'"
        )
        lmp.variable("thermo_callback python thermo_callback")
        lmp.thermo_style("custom step temp press pe vol v_thermo_callback")
        self.steps_per_thermo = steps_per_thermo
        self.thermo_per_cycle = thermo_per_cycle
        self.steps_per_cycle = steps_per_thermo * thermo_per_cycle
        log.info(
            f"Time[{self.unit_names['time']}] per step: {timestep}"
            f"  thermo: {self.steps_per_thermo * timestep}"
            f"  cycle: {self.steps_per_cycle * timestep}"
        )
        
        # Set up external force callback and density collection:
        self.force_callback = ForceCallback(
            potential=potential,
            geometry_type=geometry_type,
            lps=lps,
            dr=dr,
            n_atom_types=n_atom_types,
            potential_type=potential_type,
            pe_collect_interval=pe_collect_interval,
        )
        lmp.fix("ext all external pf/callback 1 1")
        lps.set_fix_external_callback("ext", self.force_callback, lps)
        self.curr_density = np.zeros_like(self.force_callback.hist.hist)
        self.curr_velocity = np.zeros_like(self.force_callback.hist.hist)
        
        # Prepare for thermo data collection:
        self.i_thermo = -1  # index of current thermo entry within cycle
        self.i_cycle = 0  # index of current cycle
        self.cycle_stats = np.zeros(4)  # cumulative T, P, PE, vol

        if self.enable_gcmc:
            T_for_gcmc = self.T if gcmc_temp is None else gcmc_temp
            # id gid gcmc invoke-every-N avg-X-exchanges avg-M-moves type seed T mu displace region
            self.lmp.command(
                f"fix gcmc1 all gcmc 1 10 0 1 {seed} 2 {gcmc_mu1} 0.0 region {gcmc_region1}"
            )
            self.lmp.command(
                f"fix gcmc2 all gcmc 1 10 0 1 {seed} 2 {gcmc_mu2} 0.0 region {gcmc_region2}"
            )
            self.lmp.command(f"variable n_atoms1 equal count(all,{gcmc_region1})")
            self.lmp.command(f"variable n_atoms2 equal count(all,{gcmc_region2})")


    @property
    def density(self) -> np.ndarray:
        """Density profiles, averaged over all completed cycles."""
        assert(self.i_cycle)  # Need at least one cycle before getting density
        return self.curr_density

    @property
    def velocity(self) -> np.ndarray:
        assert(self.i_cycle)
        return self.curr_velocity

    def reset_vext(self, potential: Potential) -> None:
        # make external callback every 1 step, apply forces every 1 step
        self.force_callback.set_potential(potential)

    def dump(self, name) -> None:
        self.lmp.command(f"dump {name} all movie 1000 {name}.mp4 type type size 640 480 axes yes 1 0.01")

    def undump(self, name) -> None:
        self.lmp.command(f"undump {name}")

    def reset_stats(self) -> None:
        """Reset counts / histograms."""
        # Resetting timestep conflicts with time-dependent fixes like GCMC
        if not getattr(self, "enable_gcmc", False):
            self.lmp.reset_timestep(0)
        self.i_thermo = -1
        self.i_cycle = 0
        self.curr_density.fill(0.)
        self.curr_velocity.fill(0.)
        self.force_callback.reset_stats()
        self.force_callback.reset_history()

    def run(self, n_cycles: int, run_name: str, out_filename: str = "") -> None:
        """
        Run `n_cycles` cycles of `thermo_per_cycle` x `steps_per_thermo` steps each.
        Use `run_name` to report the start and end of the run (to ease parsing the log).
        If `out_filename` is specified, save density response (HDF5) after every cycle.
        """
        log.info(f"Starting {run_name}")
        if self.enable_gcmc:
            log.info(
                f"Cycle T[{self.unit_names['temperature']}]"
                f" P[{self.unit_names['pressure']}] PE[{self.unit_names['energy']}]"
                f" vol[{self.unit_names['distance']}^3] t_cpu[s] atoms1 atoms2"
            )
        else:
            log.info(
                f"Cycle T[{self.unit_names['temperature']}]"
                f" P[{self.unit_names['pressure']}] PE[{self.unit_names['energy']}]"
                f" vol[{self.unit_names['distance']}^3] t_cpu[s]"
            )

        # self.dump(run_name)

        # Optionally prune side regions at run start
        if self.prune_regions:
            try:
                self.lmp.delete_atoms("region atom_box3")
                self.lmp.delete_atoms("region atom_box2")
            except Exception:
                pass

        for i_cycle in range(n_cycles):
            self.lmp.run(self.steps_per_cycle)
            if out_filename:
                self.save_response(out_filename + f".{i_cycle:05d}.h5")
        log.info(f"Completed {run_name}")

        # self.undump(run_name)

    def __call__(self, unused_lmp_ptr) -> float:
        """Callback function invoked during each thermo cycle to collect densities."""
        if self.i_thermo == -1:
            # Ignore first step redundant with previous cycle:
            self.i_thermo = 0
            return 0.
        
        # Collect results over cycle:
        if self.force_callback.geometry_type == Planar:
            pressure = self.lps.get_thermo("pzz")
        elif self.force_callback.geometry_type == Cylindrical:
            pressure = 0.5 * (self.lps.get_thermo("pxx") + self.lps.get_thermo("pyy"))
        else:
            pressure = self.lps.get_thermo("press")
        self.cycle_stats += np.stack((
            self.lps.get_thermo("temp"),
            pressure,
            self.lps.get_thermo("pe"),
            self.lps.get_thermo("vol"),
        ))
        self.i_thermo += 1

        # Report when cycle complete:
        if self.i_thermo == self.thermo_per_cycle:
            cycle_norm = 1. / self.thermo_per_cycle
            T, P, PE, vol = self.cycle_stats * cycle_norm
            t_cpu = time.time() - self.t_start
            self.curr_density = self.force_callback.density
            self.curr_velocity = self.force_callback.velocity_field
            self.i_cycle += 1
            if self.enable_gcmc:
                atoms1 = self.lps.extract_variable('n_atoms1')
                atoms2 = self.lps.extract_variable('n_atoms2')
                log.info(
                    f"{self.i_cycle:^5d} {T:7.3f} {P:7.3f} {PE:^12.3f} "
                    f"{vol:^8.1f} {t_cpu:7.1f} {atoms1} {atoms2}"
                )
            else:
                log.info(
                    f"{self.i_cycle:^5d} {T:7.3f} {P:7.3f} {PE:^12.3f} "
                    f"{vol:^8.1f} {t_cpu:7.1f}"
                )
            # Reset within-cycle quantities:
            self.force_callback.reset_stats()
            self.cycle_stats.fill(0.)
            self.i_thermo = -1
        
        return 0.

    def save_response(self, filename: str) -> None:
        """Save averaged densities and potentials to file."""
        if self.is_head:
            with h5py.File(filename, "w") as fp:
                fp["r"] = self.force_callback.r
                fp["n"] = self.density
                fp["V"] = self.force_callback.get_potential()
                fp["vz"] = self.velocity
                fp.attrs["T"] = self.T
                if self.P is not None:
                    fp.attrs["P"] = self.P
                fp.attrs["geometry"] = self.force_callback.geometry_type.__name__
                self.force_callback.save(fp)
