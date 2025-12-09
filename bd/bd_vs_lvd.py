import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load data
with h5py.File("results/2025-12-08-test-old-open/trials/0/collection.00000.h5", "r") as f_old:
    r_old = f_old["r"][:]
    n_old = f_old["n"][:]

with h5py.File("results/2025-12-09-test-new-open/trials/0/collection.00000.h5", "r") as f_new:
    r_new = f_new["r"][:]
    n_new = f_new["n"][:]
    vz_new = f_new["vz"][:]  # This is the new field you added

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

# 1. DENSITY CHECK (Should match perfectly)
ax[0].plot(r_old, n_old, 'k--', label="Old (NVT)")
ax[0].plot(r_new, n_new, 'r-', alpha=0.7, label="New (Langevin)")
ax[0].set_ylabel("Density")
ax[0].legend()
ax[0].set_title("Check 1: Do thermodynamics match?")

# 2. VELOCITY CHECK
# If equilibrium: Should be noise around 0.
# If flowing: Should show a profile, NOT just follow density shape.
ax[1].plot(r_new, vz_new, 'b-')
ax[1].axhline(0, color='k', linestyle=':', alpha=0.5)
ax[1].set_ylabel("Velocity $v_z$")
ax[1].set_xlabel("Position $z$")
ax[1].set_title("Check 2: Is velocity normalized correctly?")

plt.tight_layout()
plt.savefig("compare_results.png")