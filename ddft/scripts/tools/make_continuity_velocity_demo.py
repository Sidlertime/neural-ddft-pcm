"""
Builds a presentation/report demo combining two earlier diagnostic threads
into one comparison:
  1. "Continuity (true v)": rolls the true MD density forward using the
     continuity equation driven by the TRUE MD velocity (same methodology
     as continuity_diagnostic.py's "substepped" variant) - shows how well
     continuity-alone (no momentum equation, no gamma/eta/m/kBT) explains
     the MD's own density evolution.
  2. "Corrected velocity": pairs the TRUE MD density with the already-
     derived "PDE-consistent" velocity (derive_pde_velocity.py's output,
     results/diagnostics/pde_velocity_3a/v_derived.npy) - the velocity
     that, by construction, reproduces the MD density via continuity.

Saves both as standalone pseudo-run directories in the project's standard
rho*.npy/v*.npy per-frame convention, so the existing plot_movie_multiple /
plot_sliced_snapshots infrastructure can be reused directly - no new
plotting code needed.
"""
import os
import sys
import builtins

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phi.torch.flow import *  # noqa: F401,F403 - needed for CenteredGrid/Box/etc below
# from phi.torch.flow import * shadows Python's builtin round() with PhiFlow's
# own (single-argument) round_ - use builtins.round below (same fix as
# continuity_diagnostic.py / force_factory.py).
from neural_ddft.data.dataset import MDWindowDataset
from continuity_diagnostic import rollout_one_frame  # noqa: E402

MD_PATH = "../data/lj_data/ref_3a_langevin_01/avg_2000/"
FRAME_INTERVAL = 0.01
DT = 1e-4
DIFFUSION_COEFF = 0.0  # matches the already-derived velocity's own assumption

OUT_CONTINUITY = "results/diagnostics/md_continuity_demo/continuity_true_v"
OUT_CORRECTED = "results/diagnostics/md_continuity_demo/corrected_velocity"


def main():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    os.chdir(repo_root)

    ds = MDWindowDataset(MD_PATH, window_size=1)
    rho_traj = ds.rho_traj.numpy()
    v_traj = ds.v_traj.numpy()
    dx = ds.dr
    box_length = ds.box_length
    domain_res = ds.domain_resolution
    T = rho_traj.shape[0]
    n_substeps = max(1, builtins.round(FRAME_INTERVAL / DT))

    v_derived = np.load("results/diagnostics/pde_velocity_3a/v_derived.npy")
    assert v_derived.shape == v_traj.shape, f"{v_derived.shape} vs {v_traj.shape}"

    os.makedirs(OUT_CONTINUITY, exist_ok=True)
    os.makedirs(OUT_CORRECTED, exist_ok=True)

    print(f"Rolling continuity-only density forward using true MD velocity ({T} frames)...")
    rho_current = CenteredGrid(
        math.tensor(rho_traj[0].astype(np.float32), spatial('x')),
        extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length),
    )
    # Frame 0: save the true initial condition itself (no rollout yet)
    np.save(f"{OUT_CONTINUITY}/rho{0*FRAME_INTERVAL:06.3f}.npy", rho_traj[0])
    np.save(f"{OUT_CONTINUITY}/v{0*FRAME_INTERVAL:06.3f}.npy", v_traj[0])
    np.save(f"{OUT_CORRECTED}/rho{0*FRAME_INTERVAL:06.3f}.npy", rho_traj[0])
    np.save(f"{OUT_CORRECTED}/v{0*FRAME_INTERVAL:06.3f}.npy", v_derived[0])

    for t in range(T - 1):
        rho_next_grid = rollout_one_frame(
            rho_current, v_traj[t], DT, n_substeps, domain_res, box_length, DIFFUSION_COEFF
        )
        rho_next = rho_next_grid.values.native('x').cpu().numpy()
        time_val = (t + 1) * FRAME_INTERVAL

        np.save(f"{OUT_CONTINUITY}/rho{time_val:06.3f}.npy", rho_next)
        np.save(f"{OUT_CONTINUITY}/v{time_val:06.3f}.npy", v_traj[t + 1])

        # "Corrected velocity" source: true MD density (not the rolled
        # prediction) paired with the derived velocity - this pairing is
        # exactly what the derivation is FOR (by construction, this
        # velocity reproduces the true density via continuity).
        np.save(f"{OUT_CORRECTED}/rho{time_val:06.3f}.npy", rho_traj[t + 1])
        np.save(f"{OUT_CORRECTED}/v{time_val:06.3f}.npy", v_derived[t + 1])

        rho_current = rho_next_grid

    print(f"Saved {T} frames to {OUT_CONTINUITY}/ and {OUT_CORRECTED}/")


if __name__ == "__main__":
    main()
