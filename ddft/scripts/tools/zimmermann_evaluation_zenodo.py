import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from neural_ddft.models.zimmermann_net import ZimmermannNet

def extract_sliding_windows(rho, v_x, v_y, v_z, window_size=561):
    """
    Transforms 1D periodic profiles of length N into N sliding windows of length 561.
    """
    # Stack into [Channels=4, Length=N]
    x_full = torch.stack([rho, v_x, v_y, v_z], dim=0)
    
    # Calculate padding (280 on each side for a 561 window)
    pad_size = window_size // 2
    x_padded = F.pad(x_full.unsqueeze(0), (pad_size, pad_size), mode='circular').squeeze(0)
    
    # Extract windows: Shape becomes (Channels=4, N, WindowSize)
    windows = x_padded.unfold(1, window_size, 1)
    
    # Permute to PyTorch Conv1d format: [Batch=N, Channels=4, Length=561]
    return windows.permute(1, 0, 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix = "../../" if os.getcwd().endswith("tools") else "./"
    
    # 1. Load the translated PyTorch model
    model_path = f"{prefix}models/zimmermann_pytorch.pt"
    model = ZimmermannNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load the Zenodo .txt file
    data_path = f"{prefix}../data/zimmermann_data/Sim_Data-100.txt"  # Point to one of the text files
    print(f"Loading data from: {data_path}")
    
    try:
        # np.loadtxt automatically handles tabs/spaces. skiprows=1 ignores the header.
        data = np.loadtxt(data_path, skiprows=1)
        
        # Unpack columns according to the text file header:
        # z, rho, v_x, v_y, v_z, fint_x, fint_y, fint_z, fext_x, fext_y, fext_z
        z_axis = data[:, 0]
        rho_np = data[:, 1]
        vx_np, vy_np, vz_np = data[:, 2], data[:, 3], data[:, 4]
        true_fx, true_fy, true_fz = data[:, 5], data[:, 6], data[:, 7]
        
    except FileNotFoundError:
        print(f"Could not find {data_path}. Please check the path.")
        return

    # 3. Convert to PyTorch tensors
    rho = torch.tensor(rho_np, dtype=torch.float32).to(device)
    v_x = torch.tensor(vx_np, dtype=torch.float32).to(device)
    v_y = torch.tensor(vy_np, dtype=torch.float32).to(device)
    v_z = torch.tensor(vz_np, dtype=torch.float32).to(device)

    # 4. Prepare sliding windows and Predict
    batch_input = extract_sliding_windows(rho, v_x, v_y, v_z, window_size=561)
    
    with torch.no_grad():
        # Predict all spatial points simultaneously
        f_pred = model(batch_input).cpu().numpy() # Shape: (1000, 3)
        
    pred_fx = f_pred[:, 0]
    pred_fy = f_pred[:, 1]
    pred_fz = f_pred[:, 2]

    # 5. Plot the Results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("PyTorch Translation vs Zenodo Ground Truth", fontsize=14)
    
    axes[0].plot(z_axis, true_fx, 'k--', label="Ground Truth (MD)")
    axes[0].plot(z_axis, pred_fx, 'r-', alpha=0.7, label="PyTorch Prediction")
    axes[0].set_ylabel("f_int, x")
    axes[0].legend()
    
    axes[1].plot(z_axis, true_fy, 'k--')
    axes[1].plot(z_axis, pred_fy, 'g-', alpha=0.7)
    axes[1].set_ylabel("f_int, y")
    
    axes[2].plot(z_axis, true_fz, 'k--')
    axes[2].plot(z_axis, pred_fz, 'b-', alpha=0.7)
    axes[2].set_ylabel("f_int, z")
    axes[2].set_xlabel("z / sigma")
    
    plt.tight_layout()
    plt.savefig("zimmermann_verification.png", dpi=300)
    print("Verification plot saved to zimmermann_verification.png")

if __name__ == "__main__":
    main()