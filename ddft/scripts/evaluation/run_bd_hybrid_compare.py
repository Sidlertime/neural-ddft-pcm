"""
Runs the legacy inertial-PhiFlow DDFT solver (ddft_neural_closed_v_phiflow.py)
against all 5 BD/closed datasets in two modes - equilibrium (--disable_pft,
PCM free-energy functional only, no learned correction) and hybrid (with the
historical superadiabatic_net_v0.ckpt correction active) - for the
"hybrid vs equilibrium vs pcm" three-way comparison (the third leg, "pcm",
is the genuine overdamped Smoluchowski solve already computed separately via
ddft_neural_closed.py / the original ddft-c2-closed data).
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_CONCURRENT = 10

POTENTIALS = {
    "gaussian": "../data/g1-closed-dz100/avg_5000",
    "3a": "../data/g28-closed-dz100/avg_5000",
    "3b": "../data/g3-closed-dz100/avg_5000",
    "3c": "../data/g4-closed-dz100/avg_5000",
    "3d": "../data/g14-closed-dz100/avg_5000",
}


def main():
    os.chdir(REPO_ROOT)
    os.makedirs("logs/bd_hybrid_compare", exist_ok=True)

    jobs = []
    for name, md_path in POTENTIALS.items():
        jobs.append((f"{name}_eq", md_path, ["--disable_pft"]))
        jobs.append((f"{name}_hybrid", md_path, ["--sup_model", "superadiabatic_net_v0.ckpt"]))

    print(f"Total jobs: {len(jobs)}")
    running = []
    completed = 0
    idx = 0

    while idx < len(jobs) or running:
        while len(running) < MAX_CONCURRENT and idx < len(jobs):
            run_id, md_path, extra_flags = jobs[idx]
            log_path = f"logs/bd_hybrid_compare/{run_id}.log"
            cmd = [
                "python3", "legacy/scripts/evaluation/ddft_neural_closed_v_phiflow.py",
                "--run_id", run_id, "--md_path", md_path, "--type", "c2",
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

    print("All BD hybrid-compare runs complete.")


if __name__ == "__main__":
    main()
