import glob
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

class MDWindowDataset(Dataset):
    def __init__(self, md_path, window_size, stride=1, sigma_smooth=1.5):
        """
        Loads the MD trajectory and creates sliding windows of length `window_size`.
        Returns batches of shape: (window_size + 1, spatial_resolution)
        """
        self.window_size = window_size
        self.stride = stride
        
        file_pattern = f"{md_path}/collection.*.h5"
        md_file_list = sorted(glob.glob(file_pattern))
        
        if len(md_file_list) < window_size + 1:
            raise ValueError(f"Need at least {window_size + 1} files for this window size.")

        with h5py.File(md_file_list[0], 'r') as f:
            self.V_ext = torch.tensor(np.array(f['V']).flatten(), dtype=torch.float32)
            self.r_np = np.array(f['r']).flatten()
            self.domain_resolution = len(self.r_np)
            self.box_length = float(self.r_np[-1] - self.r_np[0])
            self.dr = float(self.r_np[1] - self.r_np[0])

        # Load and smooth data
        rho_list, v_list = [], []
        for file_path in md_file_list:
            with h5py.File(file_path, 'r') as f:
                rho_raw = np.array(f['n']).flatten()
                v_raw = np.array(f['vz']).flatten()
                
                # Apply Periodic Gaussian Smoothing
                rho_smooth = gaussian_filter1d(rho_raw, sigma=sigma_smooth, mode='wrap')
                v_smooth = gaussian_filter1d(v_raw, sigma=sigma_smooth, mode='wrap')
                
                rho_list.append(rho_smooth)
                v_list.append(v_smooth)
                
        # Stack into full trajectory tensors: Shape (Time, X)
        self.rho_traj = torch.tensor(np.stack(rho_list), dtype=torch.float32)
        self.v_traj = torch.tensor(np.stack(v_list), dtype=torch.float32)
        
        max_start_idx = len(self.rho_traj) - self.window_size
        self.starts = np.arange(0, max_start_idx, self.stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        # Return a chunk of the trajectory from start_idx to start_idx + window_size
        # The chunk length is window_size + 1 (1 start state + N target states)
        start = self.starts[idx]
        return {
            'rho': self.rho_traj[start : start + self.window_size + 1],
            'v': self.v_traj[start : start + self.window_size + 1]
        }


def potential_name_from_path(path):
    """
    Derives a human-readable potential identifier from an MD data path.
    Data is conventionally laid out as <potential_dir>/<avg_subfolder>/collection.*.h5
    (e.g. "ref_3a_langevin_01/avg_2000/") - the trailing "avg_*" subfolder is shared
    across many potentials (e.g. all of ref_3a-3d use "avg_2000"), so prefer the
    parent directory name when the immediate basename looks like such a subfolder.
    """
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    if base.startswith("avg_"):
        parent = os.path.dirname(norm)
        if parent:
            return os.path.basename(parent)
    return base


class MultiPotentialMDDataset(Dataset):
    """
    Combines per-potential MDWindowDataset instances (e.g. a Gaussian well and
    several Karnik reference potentials) into a single indexable dataset.

    Every sample is tagged with its originating potential index ('potential_id')
    so the training loop can look up the correct external potential V_ext and,
    via `phase_chunk_indices`, select one continuous trajectory chunk PER
    POTENTIAL rather than a single chunk spanning the whole concatenation
    (chunk boundaries must never straddle two different potentials' data).
    """

    @staticmethod
    def _assert_consistent_grid(names, datasets):
        ref_name, ref = names[0], datasets[0]
        for name, ds in zip(names[1:], datasets[1:]):
            if (ds.domain_resolution != ref.domain_resolution
                    or not np.isclose(ds.box_length, ref.box_length)
                    or not np.isclose(ds.dr, ref.dr)):
                raise ValueError(
                    f"Potential '{name}' grid (res={ds.domain_resolution}, "
                    f"box={ds.box_length}, dr={ds.dr}) does not match reference "
                    f"potential '{ref_name}' (res={ref.domain_resolution}, "
                    f"box={ref.box_length}, dr={ref.dr}). All potentials in a mixed "
                    "dataset must share the same spatial grid."
                )

    def __init__(self, md_paths, window_size, stride=1, sigma_smooth=1.5):
        if isinstance(md_paths, str):
            md_paths = [md_paths]
        if len(md_paths) == 0:
            raise ValueError("md_paths must contain at least one MD directory")

        self.potential_names = [potential_name_from_path(p) for p in md_paths]
        self.datasets = [
            MDWindowDataset(p, window_size, stride=stride, sigma_smooth=sigma_smooth)
            for p in md_paths
        ]

        self._assert_consistent_grid(self.potential_names, self.datasets)

        ref = self.datasets[0]
        self.domain_resolution = ref.domain_resolution
        self.box_length = ref.box_length
        self.dr = ref.dr
        # Shape: (n_potentials, domain_resolution) - indexed per-sample via 'potential_id'
        self.V_ext = torch.stack([ds.V_ext for ds in self.datasets])

        self._lengths = [len(ds) for ds in self.datasets]
        self._offsets = np.cumsum([0] + self._lengths[:-1])

    def __len__(self):
        return sum(self._lengths)

    def _locate(self, idx):
        pot_idx = int(np.searchsorted(self._offsets, idx, side='right') - 1)
        local_idx = idx - self._offsets[pot_idx]
        return pot_idx, int(local_idx)

    def __getitem__(self, idx):
        pot_idx, local_idx = self._locate(idx)
        item = self.datasets[pot_idx][local_idx]
        item['potential_id'] = pot_idx
        return item

    def phase_chunk_indices(self, chunk_fraction, rng=None):
        """
        Picks one random continuous chunk per potential (scoped to that
        potential's own trajectory, preserving the "Continuous Multi-Start
        Shifting Chunks" training strategy) and returns their combined
        global indices into this dataset.
        """
        rng = rng if rng is not None else np.random
        indices = []
        for pot_idx, length in enumerate(self._lengths):
            chunk_size = max(1, int(chunk_fraction * length))
            max_start = max(0, length - chunk_size)
            start = int(rng.randint(0, max_start + 1))
            indices.extend(int(self._offsets[pot_idx]) + np.arange(start, start + chunk_size))
        return indices