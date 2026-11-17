"""
Runs PhiFlow / Goddard / py-pde SEQUENTIALLY (not parallel - CPU contention
would distort the timing comparison) across {equilibrium, hybrid-vext-only,
hybrid-full-6-channel} on potential 3a, recording wall-clock time and
outcome (survived/blowup/NaN) to results/solver_timing_comparison.md.

PhiFlow already logs wall_clock_seconds in its own summary.json for runs
already on disk - reused directly rather than rerun. Goddard/py-pde are
freshly timed here since neither logs this internally.
"""
import json
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Sibling checkout of the pseudo-spectral solver (github.com/Sidlertime/goddard-ddft);
# override with the GODDARD_ROOT environment variable.
GODDARD_ROOT = os.environ.get("GODDARD_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "goddard")))

FULL_CHANNEL_CKPT_GODDARD = "results/single_potential_v7_v8/train/2026-11-06_train_3a_only_v7a_lw10_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"
VEXT_CKPT = "results/2026-11-06_channel_ablation/3a_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"
FULL_CHANNEL_FLAGS = ["--sup_use_vext", "--sup_use_rho_prev", "--sup_use_v_prev", "--sup_use_accel", "--sup_use_gamma", "--sup_use_eta"]
VEXT_FLAGS = ["--sup_use_vext"]

results = []


def record(solver, variant, elapsed_s, status, out_dir):
    entry = {"solver": solver, "variant": variant, "elapsed_seconds": round(elapsed_s, 1), "status": status, "out_dir": out_dir}
    results.append(entry)
    print(entry)
    with open(os.path.join(REPO_ROOT, "results/report_figures/solver_timing_raw.json"), "w") as f:
        json.dump(results, f, indent=2)


def check_goddard_status(out_dir):
    rho_files = sorted([f for f in os.listdir(out_dir) if f.startswith("rho") and f.endswith(".npy")]) if os.path.isdir(out_dir) else []
    if not rho_files:
        return "failed (no output)"
    import numpy as np
    last = np.load(os.path.join(out_dir, rho_files[-1]))
    if np.isnan(last).any():
        return f"NaN (after {len(rho_files)} frames)"
    if len(rho_files) < 395:  # expect ~400 for a full t_range=4.0 run
        return f"incomplete/blowup ({len(rho_files)} frames)"
    return "survived (full trajectory)"


def run_goddard(run_id, md_path, sup_model=None, extra_flags=None):
    cmd = ["python3", "scripts/ddft_neural.py", "--run_id", run_id, "--md_path", md_path,
           "--type", "c2", "--gamma", "0.1", "--eta", "0.6944"]
    if sup_model:
        cmd += ["--sup_model", sup_model] + (extra_flags or [])
    t0 = time.time()
    subprocess.run(cmd, cwd=GODDARD_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    # Find the most recent matching output dir
    candidates = sorted([d for d in os.listdir(os.path.join(GODDARD_ROOT, "results")) if d.endswith(f"{run_id}_01") or run_id in d])
    out_dir = os.path.join(GODDARD_ROOT, "results", candidates[-1]) if candidates else None
    status = check_goddard_status(out_dir) if out_dir else "failed (no dir)"
    record("Goddard", run_id, elapsed, status, out_dir)


def check_pypde_status(out_dir):
    rho_files = sorted([f for f in os.listdir(out_dir) if f.startswith("rho") and f.endswith(".npy")]) if os.path.isdir(out_dir) else []
    if not rho_files:
        return "failed (no output)"
    import numpy as np
    last = np.load(os.path.join(out_dir, rho_files[-1]))
    if np.isnan(last).any():
        return f"NaN (after {len(rho_files)} frames)"
    if len(rho_files) < 395:
        return f"incomplete/blowup ({len(rho_files)} frames)"
    return "survived (full trajectory)"


def run_pypde(run_id, md_path, sup_model_path=None, extra_flags=None):
    cmd = ["python3", "legacy/scripts/evaluation/pypde_eval.py", "--run_id", run_id, "--md_path", md_path,
           "--type", "c2", "--gamma", "0.1", "--eta", "0.6944", "--dt", "1e-4"]
    if sup_model_path:
        cmd += ["--sup_model_path", sup_model_path] + (extra_flags or [])
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    candidates = sorted([d for d in os.listdir(os.path.join(REPO_ROOT, "results")) if run_id in d])
    out_dir = os.path.join(REPO_ROOT, "results", candidates[-1]) if candidates else None
    status = check_pypde_status(out_dir) if out_dir else "failed (no dir)"
    record("py-pde", run_id, elapsed, status, out_dir)


def main():
    md_path_3a_goddard = "../neural-ddft-pcm/data/lj_data/ref_3a_langevin_01/avg_2000"
    md_path_3a_pypde = "../data/lj_data/ref_3a_langevin_01/avg_2000/"

    # Goddard: equilibrium, vext-only, full-channel
    run_goddard("timing_goddard_equilibrium_3a", md_path_3a_goddard)
    run_goddard("timing_goddard_vext_3a", md_path_3a_goddard, sup_model=VEXT_CKPT, extra_flags=VEXT_FLAGS)
    run_goddard("timing_goddard_full_3a", md_path_3a_goddard, sup_model=FULL_CHANNEL_CKPT_GODDARD, extra_flags=FULL_CHANNEL_FLAGS)

    # py-pde: equilibrium, vext-only, full-channel
    run_pypde("timing_pypde_equilibrium_3a", md_path_3a_pypde)
    run_pypde("timing_pypde_vext_3a", md_path_3a_pypde, sup_model_path=VEXT_CKPT, extra_flags=VEXT_FLAGS)
    run_pypde("timing_pypde_full_3a", md_path_3a_pypde, sup_model_path=FULL_CHANNEL_CKPT_GODDARD, extra_flags=FULL_CHANNEL_FLAGS)

    print("All timing runs complete.")


if __name__ == "__main__":
    main()
