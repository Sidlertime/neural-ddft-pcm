import numpy as np
import pde
import torch

# --- Mock Data ---
N = 128
grid = pde.CartesianGrid([[0, 10]], N, periodic=True)
rho = pde.ScalarField(grid, data=np.ones(N))
v = pde.VectorField(grid, data=np.zeros((1, N)))
V_ext = pde.ScalarField(grid, data=np.random.rand(N))

# --- Mock User Function ---
def excess_force(rho_data):
    # Returns shape (1, N)
    return np.ones((1, N))

# --- TEST 1: Advection ---
try:
    res = v @ v.gradient(v.grid.make_operator("gradient", bc="periodic"))
    # Or in expression language:
    print("Advection check: OK")
except Exception as e:
    print(f"Advection failed: {e}")

# --- TEST 2: Gradient of V_ext ---
try:
    res = V_ext.gradient("periodic")
    print(f"Grad(V_ext) shape: {res.data.shape}") # Should be (1, N)
except Exception as e:
    print(f"Grad(V_ext) failed: {e}")

# --- TEST 3: User Function Integration ---
try:
    func_out = excess_force(rho.data)
    print(f"User Func shape: {func_out.shape}") # Should be (1, N)
    
    # Try adding them
    total = res.data + func_out
    print("Addition check: OK")
except Exception as e:
    print(f"Addition failed: {e}")

# --- TEST 4: Pressure Term ---
try:
    # (k T / rho) * grad(rho)
    grad_rho = rho.gradient("periodic").data
    term = (1.0 / rho.data) * grad_rho
    print(f"Pressure shape: {term.shape}") # Should be (1, N)
except Exception as e:
    print(f"Pressure failed: {e}")