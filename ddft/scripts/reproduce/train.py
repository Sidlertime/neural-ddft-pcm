"""
Train the correction models of the report listed in experiments/report.yaml.
Models with an existing training run are skipped.

Examples (run from neural-ddft-pcm/ddft):
    python scripts/reproduce/train.py --dry_run
    python scripts/reproduce/train.py --only density_weighted/3a density_weighted/gaussian
    python scripts/reproduce/train.py --jobs 3          # several runs share one GPU

One model takes about 45-60 min on a GTX 1080 (80 epochs, 100 solver substeps
per MD frame). Afterwards evaluate with: evaluate.py --models trained
"""
import argparse
import fnmatch
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
    p.add_argument("--only", nargs="*", help="model names or patterns, e.g. 'density_weighted/*'")
    p.add_argument("--jobs", type=int, default=1, help="trainings in parallel")
    p.add_argument("--force", action="store_true", help="retrain models that already have a checkpoint")
    p.add_argument("--dry_run", action="store_true", help="only print the jobs")
    args = p.parse_args()

    manifest = X.load(args.manifest)
    jobs = []
    for model in manifest["models"]:
        if args.only and not any(fnmatch.fnmatch(model, pat) for pat in args.only):
            continue
        if X.trained_checkpoint(model) and not args.force:
            continue
        log = os.path.join(paths.RESULTS_ROOT, "report", "train", model + ".log")
        jobs.append((model, X.train_command(manifest, model), log))

    print(f"{len(jobs)} trainings to run")
    if args.dry_run:
        for name, cmd, _ in jobs:
            print(f"\n# {name}\n" + " ".join(cmd))
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failed = sum(r != 0 for r in pool.map(run, jobs))
    print(f"finished, {failed} failed")


if __name__ == "__main__":
    main()
