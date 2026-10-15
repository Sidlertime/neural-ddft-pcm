import h5py
import torch
import numpy as np
from neural_ddft.models.zimmermann_net import ZimmermannNet

# Load into PyTorch model
print("Translating Keras weights to PyTorch...")
model = ZimmermannNet()
file_path = 'NeuralForceFunctional.keras'

# Load Keras weights using h5py
with h5py.File(file_path, 'r') as f:
    # --- CONV LAYER 1 ---
    # Keras: (20, 4, 16) -> PyTorch: (16, 4, 20)
    w1 = f['model_weights/conv1d/conv1d/kernel:0'][:]
    b1 = f['model_weights/conv1d/conv1d/bias:0'][:]
    model.conv1.weight.data = torch.tensor(np.transpose(w1, (2, 1, 0)))
    model.conv1.bias.data = torch.tensor(b1)
    
    # --- CONV LAYER 2 ---
    # Keras: (20, 16, 32) -> PyTorch: (32, 16, 20)
    w2 = f['model_weights/conv1d_1/conv1d_1/kernel:0'][:]
    b2 = f['model_weights/conv1d_1/conv1d_1/bias:0'][:]
    model.conv2.weight.data = torch.tensor(np.transpose(w2, (2, 1, 0)))
    model.conv2.bias.data = torch.tensor(b2)
    
    # --- CONV LAYER 3 ---
    # Keras: (20, 32, 64) -> PyTorch: (64, 32, 20)
    w3 = f['model_weights/conv1d_2/conv1d_2/kernel:0'][:]
    b3 = f['model_weights/conv1d_2/conv1d_2/bias:0'][:]
    model.conv3.weight.data = torch.tensor(np.transpose(w3, (2, 1, 0)))
    model.conv3.bias.data = torch.tensor(b3)
    
    # --- DENSE LAYER 1 ---
    # Keras: (4544, 100) -> PyTorch: (100, 4544)
    w4 = f['model_weights/dense/dense/kernel:0'][:]
    b4 = f['model_weights/dense/dense/bias:0'][:]
    model.fc1.weight.data = torch.tensor(np.transpose(w4, (1, 0)))
    model.fc1.bias.data = torch.tensor(b4)
    
    # --- DENSE LAYER 2 (FINT) ---
    # Keras: (100, 3) -> PyTorch: (3, 100)
    w5 = f['model_weights/FINT/FINT/kernel:0'][:]
    b5 = f['model_weights/FINT/FINT/bias:0'][:]
    model.fc2.weight.data = torch.tensor(np.transpose(w5, (1, 0)))
    model.fc2.bias.data = torch.tensor(b5)
    
save_path = 'zimmermann_pytorch.pt'
torch.save(model.state_dict(), save_path)
print(f"PyTorch model saved to {save_path}")

# Pure NumPy Forward Pass (Independent Ground Truth)
def numpy_forward(x_np, f):
    """
    Evaluates the network strictly using NumPy matrix operations 
    using the raw Keras weights from the HDF5 file.
    x_np shape: (1, 561, 4) [Keras format: Batch, Length, Channels]
    """
    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    curr = x_np
    
    # 3 Conv Blocks
    for name, k_shape in [('conv1d', (20, 4, 16)), 
                          ('conv1d_1', (20, 16, 32)), 
                          ('conv1d_2', (20, 32, 64))]:
        w = f[f'model_weights/{name}/{name}/kernel:0'][:]
        b = f[f'model_weights/{name}/{name}/bias:0'][:]
        
        # Keras 'same' padding (left=9, right=10)
        curr = np.pad(curr, ((0, 0), (9, 10), (0, 0)), mode='constant')
        
        # 1D Convolution
        L_out = curr.shape[1] - 20 + 1
        out = np.zeros((curr.shape[0], L_out, k_shape[2]), dtype=np.float32)
        for i in range(L_out):
            window = curr[:, i:i+20, :] # (Batch, 20, Cin)
            out[:, i, :] = np.tensordot(window, w, axes=([1, 2], [0, 1])) + b
        
        curr = softplus(out)
        
        # Avg Pool (pool_size=2, stride=2, padding='same')
        if curr.shape[1] % 2 != 0:
            curr = np.pad(curr, ((0, 0), (0, 1), (0, 0)), mode='constant')
        curr = (curr[:, 0::2, :] + curr[:, 1::2, :]) / 2.0

    # Flatten (Row-Major / C-contiguous as Keras does)
    flat = curr.reshape(curr.shape[0], -1)
    
    # Dense 1
    w_d1 = f['model_weights/dense/dense/kernel:0'][:]
    b_d1 = f['model_weights/dense/dense/bias:0'][:]
    h = softplus(np.dot(flat, w_d1) + b_d1)
    
    # Dense 2 (FINT)
    w_d2 = f['model_weights/FINT/FINT/kernel:0'][:]
    b_d2 = f['model_weights/FINT/FINT/bias:0'][:]
    out = np.dot(h, w_d2) + b_d2
    return out

# Test Execution and Comparison
# Create identical dummy inputs
np.random.seed(42)
dummy_keras = np.random.randn(1, 561, 4).astype(np.float32) # (Batch, Length, Channels)

# PyTorch format: (Batch, Channels, Length)
dummy_torch = torch.tensor(np.transpose(dummy_keras, (0, 2, 1)))

# Evaluate both
print("Computing NumPy ground-truth forward pass...")
with h5py.File(file_path, 'r') as f:
    np_out = numpy_forward(dummy_keras, f)

print("Computing PyTorch forward pass...")
model.eval()
with torch.no_grad():
    pt_out = model(dummy_torch).numpy()

# Compare
max_diff = np.max(np.abs((np_out - pt_out)/np_out)) # as percentage
print("\n" + "="*50)
print(f"NumPy Output:   {np_out[0]}")
print(f"PyTorch Output: {pt_out[0]}")
print(f"Max Absolute Discrepancy: {max_diff:.2e}")

if max_diff < 1e-4:
    print("VERIFICATION SUCCESS: PyTorch translation is identical to Keras logic!")
else:
    print("VERIFICATION FAILED: Discrepancy detected between frameworks.")
print("="*50)
