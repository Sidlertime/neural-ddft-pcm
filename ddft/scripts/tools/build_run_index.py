"""
build_run_index.py
On-demand scanner over results/*/ that reports what each run was and how it
went, without opening 40+ folders by hand. Derives everything from each run's
own config.json/summary.json/training_log.csv at scan time (no ledger file to
go stale) - safe to run anytime, including against runs older than this tool.
"""
import os
import csv
import json
import glob
import argparse


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _count_csv_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header


def _md_paths_repr(config):
    if "md_paths" in config:
        return ", ".join(config["md_paths"])
    return config.get("md_path", "")


def scan_results(results_dir="results"):
    """
    Scans results/*/config.json (+ optional summary.json / training_log.csv)
    and returns a list of row-dicts, one per run folder, tolerant of missing
    files (most pre-existing runs predate summary.json / per-potential CSV columns).
    """
    rows = []
    for folder in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(folder):
            continue
        config = _read_json(os.path.join(folder, "config.json"))
        if config is None:
            continue  # not a run folder (e.g. stray files, "archive/")

        summary = _read_json(os.path.join(folder, "summary.json"))
        training_log_path = os.path.join(folder, "training_log.csv")
        has_training_log = os.path.exists(training_log_path)
        run_type = "train" if "curriculum" in config else "eval"

        row = {
            "folder": os.path.basename(folder),
            "run_type": run_type,
            "model_mode": config.get("model_mode"),
            "md_path_or_paths": _md_paths_repr(config),
            "has_summary": summary is not None,
            "blowup_occurred": summary.get("blowup", {}).get("occurred") if summary else None,
            "final_density_mse": (
                summary.get("reference_comparison", {}).get("density_metrics", {}).get("overall_mse")
                if summary and summary.get("reference_comparison") else None
            ),
            "force_ratio_mean": summary.get("force_ratio", {}).get("mean_over_trajectory") if summary else None,
            "n_epochs_completed": _count_csv_rows(training_log_path) if has_training_log else None,
        }
        rows.append(row)
    return rows


def _fmt(value, width):
    if value is None:
        s = "-"
    elif isinstance(value, float):
        s = f"{value:.4e}"
    else:
        s = str(value)
    return s[:width].ljust(width)


def print_table(rows):
    columns = ["folder", "run_type", "model_mode", "has_summary", "blowup_occurred",
               "final_density_mse", "force_ratio_mean", "n_epochs_completed"]
    widths = {"folder": 42, "run_type": 8, "model_mode": 11, "has_summary": 11,
              "blowup_occurred": 15, "final_density_mse": 18, "force_ratio_mean": 16,
              "n_epochs_completed": 10}
    header = " | ".join(_fmt(c, widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(_fmt(row[c], widths[c]) for c in columns))


def main():
    parser = argparse.ArgumentParser(description="Scan results/ and summarize every run.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--mode", type=str, default=None, help="Filter by model_mode (equilibrium/hybrid/zimmermann)")
    parser.add_argument("--run_type", type=str, default=None, choices=["train", "eval"])
    parser.add_argument("--format", type=str, default="table", choices=["table", "json"])
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON output (format=json)")
    args = parser.parse_args()

    rows = scan_results(args.results_dir)
    if args.mode:
        rows = [r for r in rows if r["model_mode"] == args.mode]
    if args.run_type:
        rows = [r for r in rows if r["run_type"] == args.run_type]

    if args.format == "json":
        output = json.dumps(rows, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(output)
            print(f"Wrote {len(rows)} rows to {args.out}")
        else:
            print(output)
    else:
        print_table(rows)
        print(f"\n{len(rows)} run(s) found in {args.results_dir}/")


if __name__ == "__main__":
    main()
