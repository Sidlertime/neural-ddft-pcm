"""
Tier-2 confirmation pass at full t_range=4.0 for the channel-ablation grid.
Scoped to 5 priority combos (not all 16, to fit today's deadline):
  - grad_hist_accel_vext: best Tier-1 self (3a) density RMSE (0.0194)
  - accel_vext:           best Tier-1 cross (3d zero-shot) density RMSE (0.0673)
  - accel:                near-best on BOTH self (0.0213) and cross (0.0681)
                           with only 1 extra channel group - interesting minimal-cost case
  - min_vext:             true middle-of-pack on both rankings (self rank 9/16, cross rank 8/16)
  - hist2:                another middle-ish combo, different channel group (history, no accel)
                           (self rank 13/16, cross rank 7/16)
Self + cross3d each -> 10 eval jobs total.
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 3

CHECKPOINTS = {
    "grad_hist_accel_vext": ("configs/channel_ablation_16/grad_hist_accel_vext.yaml", "results/2026-11-12_channel_ablation_16/grad_hist_accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "accel_vext": ("configs/channel_ablation_16/accel_vext.yaml", "results/2026-11-12_channel_ablation_16/accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "accel": ("configs/channel_ablation_16/accel.yaml", "results/2026-11-12_channel_ablation_16/accel_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "min_vext": ("configs/channel_ablation_16/min_vext.yaml", "results/2026-11-12_channel_ablation_16/min_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "hist2": ("configs/channel_ablation_16/hist2.yaml", "results/2026-11-12_channel_ablation_16/hist2_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
}

TARGETS = {
    "self": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "cross3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
}


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("logs/channel_ablation_tier2", exist_ok=True)

    jobs = []
    for name, (config, ckpt) in CHECKPOINTS.items():
        for target_name, md_path in TARGETS.items():
            jobs.append((name, target_name, config, ckpt, md_path))

    print(f"Total jobs: {len(jobs)}")
    running = []
    completed = 0
    idx = 0

    while idx < len(jobs) or running:
        while len(running) < MAX_CONCURRENT and idx < len(jobs):
            name, target_name, config, ckpt, md_path = jobs[idx]
            run_id = f"channel_ablation_16_eval_tier2/{name}_{target_name}"
            log_path = f"logs/channel_ablation_tier2/{name}_{target_name}.log"
            cmd = [
                "python3", "scripts/evaluation/phiflow_eval.py",
                "--config", config,
                "--run_id", run_id,
                "--sup_model_path", ckpt,
                "--md_path", md_path,
                "--t_range", "4.0",
            ]
            log_f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((proc, log_f, name, target_name))
            print(f"[{idx+1}/{len(jobs)}] launched {name} / {target_name}")
            idx += 1

        time.sleep(10)
        still_running = []
        for proc, log_f, name, target_name in running:
            if proc.poll() is None:
                still_running.append((proc, log_f, name, target_name))
            else:
                log_f.close()
                completed += 1
                print(f"  [done {completed}/{len(jobs)}] {name} / {target_name} exit={proc.returncode}")
        running = still_running

    print("All Tier-2 evals complete.")


if __name__ == "__main__":
    main()
