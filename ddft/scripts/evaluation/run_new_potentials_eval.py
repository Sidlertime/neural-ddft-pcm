"""
Eval sweep for the 4 new potentials (mg1, mg2, s1 trained; s2 held out
zero-shot). Jobs: self-eval for mg1/mg2/s1, zero-shot cross-eval of each
onto s2, and equilibrium (no-correction) baselines for all 4 - all at
t_range=4.0, gamma=0.1/eta=0.6944 (production values, all 4 generated at
Tdamp=10).
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 4

TRAINED = {
    "mg1": "results/2026-11-15_mg1_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt",
    "mg2": "results/2026-11-15_mg2_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt",
    "s1": "results/2026-11-15_s1_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt",
}

POTENTIALS = ["mg1", "mg2", "s1", "s2"]


def md_path(p):
    return f"../data/lj_data/ref_{p}_langevin_01/avg_500/"


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("logs/new_potentials", exist_ok=True)

    jobs = []
    # self evals
    for p, ckpt in TRAINED.items():
        jobs.append((f"new_potentials_eval/{p}_self", ["--model_mode", "hybrid", "--sup_model_path", ckpt, "--md_path", md_path(p)]))
    # zero-shot cross onto s2
    for p, ckpt in TRAINED.items():
        jobs.append((f"new_potentials_eval/{p}_to_s2", ["--model_mode", "hybrid", "--sup_model_path", ckpt, "--md_path", md_path("s2")]))
    # equilibrium baselines, all 4
    for p in POTENTIALS:
        jobs.append((f"new_potentials_eval/{p}_equilibrium", ["--model_mode", "equilibrium", "--md_path", md_path(p), "--diffusion_coeff", "0.5"]))

    print(f"Total jobs: {len(jobs)}")
    running = []
    completed = 0
    idx = 0

    while idx < len(jobs) or running:
        while len(running) < MAX_CONCURRENT and idx < len(jobs):
            run_id, extra_flags = jobs[idx]
            log_path = f"logs/new_potentials/{run_id.replace('/', '_')}.log"
            cmd = [
                "python3", "scripts/evaluation/phiflow_eval.py",
                "--run_id", run_id, "--base_model_path", "models/c2_dz32.ckpt",
                "--gamma", "0.1", "--eta", "0.6944", "--dt", "1e-4", "--t_range", "4.0",
            ] + extra_flags
            log_f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((proc, log_f, run_id))
            print(f"[{idx+1}/{len(jobs)}] launched {run_id}")
            idx += 1

        time.sleep(10)
        still_running = []
        for proc, log_f, run_id in running:
            if proc.poll() is None:
                still_running.append((proc, log_f, run_id))
            else:
                log_f.close()
                completed += 1
                print(f"  [done {completed}/{len(jobs)}] {run_id} exit={proc.returncode}")
        running = still_running

    print("All new-potentials evals complete.")


if __name__ == "__main__":
    main()
