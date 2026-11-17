"""
Tier-1 fast eval pass for the 16-model channel-ablation grid: self (3a) and
cross-target (3d, zero-shot) at t_range=1.5, 32 cells total. Runs up to
MAX_CONCURRENT subprocesses at once (GPU has headroom, but be conservative
given other background work this session).
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 3

# combo_name -> (config_path, checkpoint_path)
CHECKPOINTS = {
    "min": ("configs/channel_ablation_16/min.yaml", "results/2026-11-12_channel_ablation_16/min_hybrid_03/checkpoints/hybrid_epoch_160.ckpt"),
    "grad": ("configs/channel_ablation_3a_base.yaml", "results/2026-11-06_channel_ablation/3a_base_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "hist2": ("configs/channel_ablation_16/hist2.yaml", "results/2026-11-12_channel_ablation_16/hist2_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_hist": ("configs/channel_ablation_16/grad_hist.yaml", "results/2026-11-12_channel_ablation_16/grad_hist_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "accel": ("configs/channel_ablation_16/accel.yaml", "results/2026-11-12_channel_ablation_16/accel_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_accel": ("configs/channel_ablation_16/grad_accel.yaml", "results/2026-11-12_channel_ablation_16/grad_accel_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "hist_accel": ("configs/channel_ablation_16/hist_accel.yaml", "results/2026-11-12_channel_ablation_16/hist_accel_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_hist_accel": ("configs/channel_ablation_3a_hist.yaml", "results/2026-11-06_channel_ablation/3a_hist_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "min_vext": ("configs/channel_ablation_16/min_vext.yaml", "results/2026-11-12_channel_ablation_16/min_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_vext": ("configs/channel_ablation_3a_vext.yaml", "results/2026-11-06_channel_ablation/3a_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "hist2_vext": ("configs/channel_ablation_16/hist2_vext.yaml", "results/2026-11-12_channel_ablation_16/hist2_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_hist_vext": ("configs/channel_ablation_16/grad_hist_vext.yaml", "results/2026-11-12_channel_ablation_16/grad_hist_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "accel_vext": ("configs/channel_ablation_16/accel_vext.yaml", "results/2026-11-12_channel_ablation_16/accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_accel_vext": ("configs/channel_ablation_16/grad_accel_vext.yaml", "results/2026-11-12_channel_ablation_16/grad_accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "hist_accel_vext": ("configs/channel_ablation_16/hist_accel_vext.yaml", "results/2026-11-12_channel_ablation_16/hist_accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
    "grad_hist_accel_vext": ("configs/channel_ablation_16/grad_hist_accel_vext.yaml", "results/2026-11-12_channel_ablation_16/grad_hist_accel_vext_hybrid_01/checkpoints/hybrid_epoch_160.ckpt"),
}

TARGETS = {
    "self": "../data/lj_data/ref_3a_langevin_01/avg_2000/",
    "cross3d": "../data/lj_data/ref_3d_langevin_01/avg_2000/",
}


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("logs/channel_ablation_tier1", exist_ok=True)

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
            run_id = f"channel_ablation_16_eval/{name}_{target_name}"
            log_path = f"logs/channel_ablation_tier1/{name}_{target_name}.log"
            cmd = [
                "python3", "scripts/evaluation/phiflow_eval.py",
                "--config", config,
                "--run_id", run_id,
                "--sup_model_path", ckpt,
                "--md_path", md_path,
                "--t_range", "1.5",
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

    print("All Tier-1 evals complete.")


if __name__ == "__main__":
    main()
