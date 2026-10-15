import glob
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