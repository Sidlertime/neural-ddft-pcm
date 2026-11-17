"""
Eval sweep for the 4 freshly density-weighted-retrained potentials (3c, mg1,
mg2, s1) under the fixed train_phiflow.py: self-eval for each, plus
cross-eval onto the zero-shot-heldout targets (3b, 3d, s2 - never trained
this round) and a couple of cross-family pairs, for a real generalization
matrix.
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 2

TRAINED = {
    "3c": "results/2026-11-16_retrain/3c_density_weighted_hybrid_01/checkpoints/hybrid_epoch_080.ckpt",
    "mg1": "results/2026-11-16_retrain/mg1_density_weighted_hybrid_01/checkpoints/hybrid_epoch_080.ckpt",
    "mg2": "results/2026-11-16_retrain/mg2_density_weighted_hybrid_01/checkpoints/hybrid_epoch_080.ckpt",
    "s1": "results/2026-11-16_retrain/s1_density_weighted_hybrid_01/checkpoints/hybrid_epoch_080.ckpt",
}

MD = {
    "3a": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "3b": "../data/lj_data/ref_3b_langevin_01/avg_2000/",
    "3c": "../data/lj_data/ref_3c_langevin_01/avg_2000/",
    "3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
    "mg1": "../data/lj_data/ref_mg1_langevin_01/avg_500/",
    "mg2": "../data/lj_data/ref_mg2_langevin_01/avg_500/",
    "s1": "../data/lj_data/ref_s1_langevin_01/avg_500/",
    "s2": "../data/lj_data/ref_s2_langevin_01/avg_500/",
}

# (trained_model, target_potential) pairs - self evals first, then zero-shot
# heldout targets (3b/3d/s2), then a couple of cross-family bridges.
JOBS = [
    ("3c", "3c"), ("mg1", "mg1"), ("mg2", "mg2"), ("s1", "s1"),
    ("3c", "3b"), ("3c", "3d"), ("3c", "s2"),
    ("mg1", "3d"), ("mg1", "s2"),
    ("mg2", "3a"), ("mg2", "s2"),
    ("s1", "3d"), ("s1", "s2"),
]


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("logs/retrain", exist_ok=True)

    jobs = []
    for trained, target in JOBS:
        suffix = "self" if trained == target else f"to_{target}"
        run_id = f"retrain_eval/{trained}_{suffix}"
        jobs.append((run_id, TRAINED[trained], MD[target]))

    print(f"Total jobs: {len(jobs)}")
    running, completed, idx = [], 0, 0
    while idx < len(jobs) or running:
        while len(running) < MAX_CONCURRENT and idx < len(jobs):
            run_id, ckpt, md_path = jobs[idx]
            log_path = f"logs/retrain/{run_id.replace('/', '_')}.log"
            cmd = ["python3", "scripts/evaluation/phiflow_eval.py",
                   "--config", "configs/nightshift_eval_full6ch.yaml",
                   "--run_id", run_id, "--sup_model_path", ckpt, "--md_path", md_path,
                   "--gamma", "0.1", "--eta", "0.6944", "--dt", "1e-4", "--t_range", "4.0"]
            log_f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((proc, log_f, run_id))
            print(f"[{idx+1}/{len(jobs)}] launched {run_id}")
            idx += 1
        time.sleep(15)
        still = []
        for proc, log_f, run_id in running:
            if proc.poll() is None:
                still.append((proc, log_f, run_id))
            else:
                log_f.close(); completed += 1
                print(f"  [done {completed}/{len(jobs)}] {run_id} exit={proc.returncode}")
        running = still
    print("All retrain evals complete.")


if __name__ == "__main__":
    main()
