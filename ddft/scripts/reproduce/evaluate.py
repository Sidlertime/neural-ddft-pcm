"""
Run all evaluations of the report (models and PCM-only baselines) listed in
experiments/report.yaml. Finished evaluations (summary.json exists) are skipped,
so the script can be restarted at any time.

Examples (run from neural-ddft-pcm/ddft):
    python scripts/reproduce/evaluate.py --dry_run            # list all jobs
    python scripts/reproduce/evaluate.py --jobs 8             # everything, 8 in parallel
    python scripts/reproduce/evaluate.py --tables self friction
    python scripts/reproduce/evaluate.py --models trained     # use your own training runs

One evaluation simulates 4 tau (40,000 solver steps) and takes about 20-30 min on
one CPU core.
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from neural_ddft.utils import experiments as X
from neural_ddft.utils import paths


def run(job):
    name, cmd, log = job
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=paths.DDFT_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[{'done' if rc == 0 else 'FAILED, see ' + log}] {name}", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=X.DEFAULT_MANIFEST)
    p.add_argument("--tables", nargs="*", help="only the evaluations of these tables (default: all, plus figure extras)")
    p.add_argument("--models", choices=["shipped", "trained"], default="shipped",
                   help="shipped checkpoints in models/report/ (default) or your runs from train.py")
    p.add_argument("--jobs", type=int, default=1, help="evaluations in parallel (each uses one CPU core)")
    p.add_argument("--force", action="store_true", help="rerun finished evaluations")
    p.add_argument("--dry_run", action="store_true", help="only print the jobs")
    args = p.parse_args()

    manifest = X.load(args.manifest)
    d = manifest["defaults"]
    jobs = []
    for e in X.all_evaluations(manifest, args.tables):
        if X.is_done(e, manifest) and not args.force:
            continue
        jobs.append((e.name(d), X.eval_command(manifest, e, args.models), e.out_dir(d) + ".log"))

    print(f"{len(jobs)} evaluations to run")
    if args.dry_run:
        for name, cmd, _ in jobs:
            print(f"\n# {name}\n" + " ".join(cmd))
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failed = sum(r != 0 for r in pool.map(run, jobs))
    print(f"finished, {failed} failed")


if __name__ == "__main__":
    main()
