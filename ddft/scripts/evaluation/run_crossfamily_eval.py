"""
8 cross-family zero-shot eval pairs: v7a (3a-trained) onto the new
mixed-gaussian/sinusoid potentials, and mg1/s1 (new-family-trained) onto
the original Karnik potentials - bridging the two potential families for
the first time.
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 4

V7A = "results/single_potential_v7_v8/train/2026-11-06_train_3a_only_v7a_lw10_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"
MG1 = "results/2026-11-15_mg1_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"
S1 = "results/2026-11-15_s1_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"

V7A_KWARGS = ["--config", "configs/lj_sharp_v7a_zeroshot_eval.yaml"]

MD = {
    "mg1": "../data/lj_data/ref_mg1_langevin_01/avg_500/",
    "mg2": "../data/lj_data/ref_mg2_langevin_01/avg_500/",
    "s1": "../data/lj_data/ref_s1_langevin_01/avg_500/",
    "s2": "../data/lj_data/ref_s2_langevin_01/avg_500/",
    "3a": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
}

jobs = []
for target in ["mg1", "mg2", "s1", "s2"]:
    jobs.append((f"crossfamily/v7a_to_{target}", V7A_KWARGS + ["--md_path", MD[target]]))
for target in ["3a", "3d"]:
    jobs.append((f"crossfamily/mg1_to_{target}", ["--model_mode", "hybrid", "--sup_model_path", MG1, "--md_path", MD[target]]))
    jobs.append((f"crossfamily/s1_to_{target}", ["--model_mode", "hybrid", "--sup_model_path", S1, "--md_path", MD[target]]))

print(f"Total jobs: {len(jobs)}")
running, completed, idx = [], 0, 0
os.chdir(REPO_ROOT)
while idx < len(jobs) or running:
    while len(running) < MAX_CONCURRENT and idx < len(jobs):
        run_id, extra = jobs[idx]
        log_path = f"logs/crossfamily/{run_id.replace('/', '_')}.log"
        cmd = ["python3", "scripts/evaluation/phiflow_eval.py", "--run_id", run_id,
               "--base_model_path", "models/c2_dz32.ckpt", "--gamma", "0.1", "--eta", "0.6944",
               "--dt", "1e-4", "--t_range", "4.0"] + extra
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        running.append((proc, log_f, run_id))
        print(f"[{idx+1}/{len(jobs)}] launched {run_id}")
        idx += 1
    time.sleep(10)
    still = []
    for proc, log_f, run_id in running:
        if proc.poll() is None:
            still.append((proc, log_f, run_id))
        else:
            log_f.close(); completed += 1
            print(f"  [done {completed}/{len(jobs)}] {run_id} exit={proc.returncode}")
    running = still
print("All cross-family evals complete.")
