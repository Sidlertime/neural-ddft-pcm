from lammps import lammps
import numpy as np
from mdext import MPI
from .geometry import GeometryType
from .histogram import Histogram
from typing import Callable, Tuple, Sequence
from dataclasses import dataclass
import h5py
from scipy.interpolate import interp1d


Potential = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
"""Return potential and its derivative, given square of 1D coordinate as input.
Note that the derivative is also with respect to the squared-coordinate."""


class Gaussian:
    def __init__(self, U0: float, mu:float, sigma: float) -> None:
        """Gaussian potential with peak `U0` and width `sigma`."""
        self.U0 = U0
        self.sigma = sigma
        self.mu = mu
        self.mhalf_inv_sigma_sq = -0.5 / (sigma ** 2)

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = self.U0 * np.exp(self.mhalf_inv_sigma_sq * (r - self.mu)**2)
        r_grad = self.mhalf_inv_sigma_sq * E * 2 * (r - self.mu)
        return E, r_grad


class GaussianSum:
    def __init__(self, U0: Sequence[float], mu: Sequence[float], sigma: Sequence[float]) -> None:
        """Sum of `n` Gaussian potentials with peak `U0` and width `sigma`."""
        self.U0 = U0
        self.sigma = sigma
        self.mu = mu

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = np.zeros_like(r)
        r_grad = np.zeros_like(r)
        for i in range(len(self.U0)):
            exponent = (-0.5 * (r - self.mu[i])**2) / (self.sigma[i] ** 2)
            gaussian = self.U0[i] * np.exp(exponent)
            E += gaussian
            r_grad += gaussian * (-1 * (r - self.mu[i])) / (self.sigma[i] ** 2)
        return E, r_grad


class PeriodicGaussianSpikes:
    def __init__(self, n_spikes: int = 10, amplitude: float = 1.0, sigma: float = 0.2, range_min: float = 1, range_max: float = 9) -> None:
        self.n_spikes = n_spikes
        self.amplitude = amplitude
        self.sigma = sigma
        self.range_min = range_min
        self.range_max = range_max
        
        if n_spikes > 1:
            self.positions = np.linspace(range_min, range_max, n_spikes)
        else:
            self.positions = np.array([(range_min + range_max) / 2.0])
        
        self.mhalf_inv_sigma_sq = -0.5 / (sigma ** 2)
    
    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = np.zeros_like(r)
        r_grad = np.zeros_like(r)
        
        for pos in self.positions:
            dr_squared = (r - pos)**2
            gaussian = self.amplitude * np.exp(self.mhalf_inv_sigma_sq * dr_squared)
            
            E += gaussian
            
            r_grad += 2 * self.mhalf_inv_sigma_sq * (r - pos) * gaussian
            
        return E, r_grad


class RandomGaussianSum:
    def __init__(self, L=10) -> None:
        self.L = L

        self.N = np.random.randint(5, 15)
        self.Mu = np.random.uniform(0, self.L, size=self.N)
        self.Sigma = np.random.uniform(0, self.L/10, size=self.N) + self.L/100
        self.U = np.random.normal(0, 1, size=self.N)**2

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = np.zeros_like(r)
        r_grad = np.zeros_like(r)
        
        for i in range(len(self.U)):
            dr = r - self.Mu[i]
            dr = dr - self.L * np.round(dr/self.L)
            
            exponent = (-0.5 * dr**2) / (self.Sigma[i]**2)
            gaussian = self.U[i] * np.exp(exponent)
            E += gaussian
            r_grad += gaussian * (-1 * dr) / (self.Sigma[i] ** 2)
            
        return E, r_grad


class RandomWell:
    def __init__(self, L=10) -> None:
        self.L = L
        self.w = np.random.uniform(0.4, 0.9)
        self.s = np.random.uniform(40, 60)
        self.p = np.random.uniform(2, 9)

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_mod = r % self.L
        dr = r_mod - self.L/2
        abs_dr = np.abs(dr)
        half_well_width = 0.5 * self.w * self.L

        E = np.zeros_like(r)
        r_grad = np.zeros_like(r)

        mask = abs_dr > half_well_width
        delta = abs_dr[mask] - half_well_width
        norm = (1 - self.w) * self.L / 2

        E[mask] = self.s * (delta / norm) ** self.p
        r_grad[mask] = self.s * self.p * (delta / norm) ** (self.p - 1) * (1 / norm) * np.sign(dr[mask])
        
        return E, r_grad


class RandomWellGaussianSum:
    def __init__(self, L=10) -> None:
        self.L = L
        self.w = np.random.uniform(0.4, 0.9)
        self.s = np.random.uniform(5, 10)
        self.p = np.random.uniform(2, 9)

        self.N = np.random.randint(5, 10)
        self.Mu = np.random.uniform(0, self.L, size=self.N)
        self.Sigma = np.random.uniform(0, self.L/10, size=self.N) + self.L/100
        self.U = np.random.normal(0, 1, size=self.N)**2

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = np.zeros_like(r)
        r_grad = np.zeros_like(r)
        
        for i in range(len(self.U)):
            dr = r - self.Mu[i]
            dr = dr - self.L * np.round(dr/self.L)
            
            exponent = (-0.5 * dr**2) / (self.Sigma[i]**2)
            gaussian = self.U[i] * np.exp(exponent)
            E += gaussian
            r_grad += gaussian * (-1 * dr) / (self.Sigma[i] ** 2)
            
        r_mod = r % self.L
        dr = r_mod - self.L/2
        abs_dr = np.abs(dr)
        half_well_width = 0.5 * self.w * self.L

        E_w = np.zeros_like(r)
        r_grad_w = np.zeros_like(r)

        mask = abs_dr > half_well_width
        delta = abs_dr[mask] - half_well_width

        norm = (1 - self.w) * self.L / 2

        E_w[mask] = self.s * (delta / norm) ** self.p
        r_grad_w[mask] = self.s * self.p * (delta / norm) ** (self.p - 1) * (1 / norm) * np.sign(dr[mask])
        
        return E + E_w, r_grad + r_grad_w


@dataclass
class Exponential:
    """Exponential potential"""
    A: float
    rho: float
    sigma: float 

    def __call__(self, r_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = self.A * np.exp((self.sigma - np.sqrt(r_sq)) / self.rho)
        r_sq_grad = -0.5 * E / (np.sqrt(r_sq) * self.rho)
        return E, r_sq_grad


class Custom:
    def __init__(self, filename: str) -> None:
        data = np.loadtxt(filename)
        z = data[:, 0]
        U = data[:, 1]
        F = data[:, 2]
        self.U = interp1d(z, U, kind='cubic')
        self.F = interp1d(z, F, kind='cubic')

    def __call__(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = 10 * self.U(r)
        r_grad = 10 * self.F(r)
        return E, r_grad


class ForceCallback:
    """Force callback object to apply external forces and collect densities."""

    def __init__(
        self,
        *,
        potential: Potential,
        geometry_type: GeometryType,
        lps: lammps,
        dr: float,
        n_atom_types: int,
        potential_type: int,
        pe_collect_interval: int,
    ):
        self.potential = potential
        self.geometry_type = geometry_type
        self.potential_type = potential_type
        assert 0 <= potential_type <= n_atom_types
    
        # Initialize density collection with same 1D geometry as applied potential
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        self.hist = Histogram(0.0, geometry_type.r_max(L), dr, n_atom_types)
        self.vel_hist = Histogram(0.0, geometry_type.r_max(L), dr, n_atom_types)
        self.dr = dr
        self.r = self.hist.bins

        # Calculate bin-dependent weight of histograms (eg. 2 pi r for cylindrical):
        volumes = geometry_type.volume(self.r)
        w_intervals = np.zeros(len(self.r) + 1)  # with extra intervals to left & right
        w_intervals[1 : -1] = volumes[1:] - volumes[:-1]
        self.w_bins = dr / (0.5 * (w_intervals[:-1] + w_intervals[1:]))
        self.n_steps = 0
        
        # Initialize cavitation collection:
        self.pe_collect_interval = pe_collect_interval
        self.i_call = 0
        self.pe_history = []
        self.r_min_history = []

    def reset_stats(self) -> None:
        self.hist.reset()
        self.vel_hist.reset()
        self.n_steps = 0

    def reset_history(self) -> None:
        self.i_call = 0
        self.pe_history.clear()
        self.r_min_history.clear()

    def save(self, fp: h5py.File) -> None:
        if self.pe_history:
            fp["pe_history"] = np.array(self.pe_history)
        if self.r_min_history:
            fp["r_min_history"] = np.array(self.r_min_history)

    @property
    def density(self) -> np.ndarray:
        """Densities collected with same 1D geometry as applied potential."""
        result = self.hist.hist * self.w_bins[:, None] / self.n_steps
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, result)
        return result

    # NEW CODE to use Langevin Dynamics
    @property
    def velocity_field(self) -> np.ndarray:
        """True Velocity field v(z) = Momentum(z) / Count(z)."""
        
        # 1. Get Global Sum of Velocities (Momentum)
        # We copy to avoid modifying the local histogram
        momentum_sum = self.vel_hist.hist.copy()
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, momentum_sum)

        # 2. Get Global Sum of Particle Counts (Density)
        count_sum = self.hist.hist.copy()
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, count_sum)

        # 3. Calculate Velocity: v = Sum(v_i) / Sum(1)
        # Note: Volume weights and n_steps cancel out in the ratio, 
        # so we can use the raw histogram sums directly.
        
        # Avoid division by zero
        safe_counts = np.where(count_sum > 0, count_sum, 1.0)
        v_z = momentum_sum / safe_counts
        
        # Ensure empty bins are exactly zero
        v_z[count_sum == 0] = 0.0
        
        return v_z
    
    def get_potential(self) -> np.ndarray:
        """Get the potential on the histogram grid."""
        V = np.zeros_like(self.hist.hist)
        if self.potential_type:
            V[:, self.potential_type - 1]  = self.potential(self.r)[0]
        else:
            V[:, :]  = self.potential(self.r)[0][:, None]
        return V

    def set_potential(self, potential: Potential) -> None:
        """Set the potential."""
        self.potential = potential

    def __call__(
        self,
        lps: lammps,
        ntimestep: int,
        nlocal: int,
        tag: np.ndarray,
        x: np.ndarray,
        f: np.ndarray
    ) -> None:
        """
        Callback function to add force, energy and virials for external potential.
        """
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)

        pos = x
        
        # Identify atoms to apply external potential / force to:
        types = lps.numpy.extract_atom("type")

        # Get velocity from LAMMPS
        v = lps.numpy.extract_atom("v")

        # Get energies and forces (from derived class):
        f.fill(0.)
        r = pos[:, 2]
        E, r_grad = self.potential(r)
        if self.potential_type:
            mask = np.where(types == self.potential_type, 1.0, 0.0)
            E *= mask
            r_grad *= mask
        self.geometry_type.set_force(pos, r_grad, f)
        
        # Compute total energy and virial:
        Etot = MPI.COMM_WORLD.allreduce(E.sum())
        virial_mat = pos.T @ f  # 3 x 3 matrix
        virial_mat = 0.5*(virial_mat + virial_mat.T)  # symmetric tensor
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, virial_mat)
        
        # Set energy and virial globals (forces already set in-place):
        lps.fix_external_set_energy_peratom("ext", E)
        lps.fix_external_set_energy_global("ext", Etot)
        vtot = [
            virial_mat[0, 0], virial_mat[1, 1], virial_mat[2, 2],  # xx, yy, zz
            virial_mat[0, 1], virial_mat[0, 2], virial_mat[1, 2],  # xy, xz, yz
        ]
        lps.fix_external_set_virial_global("ext", vtot)

        # Collect densities for this step:
        inv_perpendicular_volume = 1.0 / self.geometry_type.perpendicular_volume(L)
        weights = np.zeros((nlocal, self.hist.n_w))
        for i_type in range(self.hist.n_w):
            weights[(types == i_type + 1), i_type] = inv_perpendicular_volume
        self.hist.add_events(r, weights)
        self.n_steps += 1
        
        # Collection velocity (z-component) for this step:
        vel_weights = np.zeros((nlocal, self.vel_hist.n_w))
        for i_type in range(self.vel_hist.n_w):
            local_types = types[:nlocal]
            type_mask = local_types == i_type + 1
            vel_weights[type_mask, i_type] = v[type_mask, 2] * inv_perpendicular_volume
        self.vel_hist.add_events(r, vel_weights)
        
        # Cavitation collection:
        if self.pe_collect_interval:
            if self.i_call % self.pe_collect_interval == 0:
                r = pos[:, 2]
                if self.potential_type:
                    r[types != self.potential_type] = np.inf
                r_min = MPI.COMM_WORLD.allreduce(np.min(r), op=MPI.MIN)
                self.r_min_history.append(r_min)
                self.pe_history.append(Etot)
            self.i_call += 1