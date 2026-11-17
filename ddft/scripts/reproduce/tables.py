"""
Compute the report tables from the evaluation runs of scripts/reproduce/evaluate.py.

Every value is the relative change of the trajectory RMSE of a model vs. the
PCM-only model on the same data: Delta = (RMSE_model - RMSE_PCM) / RMSE_PCM,
negative is better. Rows with several seeds give mean +/- standard deviation.
If a run diverged, both runs are compared over the same shortened time window
(marked with a dagger and the end time).

Examples (run from neural-ddft-pcm/ddft):
    python scripts/reproduce/tables.py                      # all tables, Markdown
    python scripts/reproduce/tables.py --tables self inputs
    python scripts/reproduce/tables.py --latex > tables.tex
"""
import argparse

import numpy as np

from neural_ddft.utils import experiments as X
from neural_ddft.utils import paths
from neural_ddft.utils.metrics import relative_change


def fmt(values, latex=False):
    pct = 100 * np.asarray(values, dtype=float)
    pm = r"\pm" if latex else "±"
    s = f"{pct.mean():+.1f}%" if len(pct) == 1 else f"{pct.mean():+.1f} {pm} {pct.std(ddof=1):.1f}%"
    return f"${s.replace('%', r'\%')}$" if latex else s


def row_values(manifest, row):
    """(d_rho list, d_v list, note) for one table row, or None if evaluations are missing."""
    d = manifest["defaults"]
    base = row.baseline.out_dir(d)
    if not X.is_done(row.baseline, manifest) or not all(X.is_done(e, manifest) for e in row.evaluations):
        return None
    md = paths.md_dir(row.target, row.baseline.gamma)
    res = [relative_change(e.out_dir(d), base, md) for e in row.evaluations]
    notes = []
    t_end = min(r["t_end"] for r in res)
    reasons = [X.stop_reason(e.out_dir(d)) for e in row.evaluations]
    if "diverged" in reasons:
        notes.append(f"† diverged, compared for t ≤ {t_end:.2f}τ")
    elif "time limit" in reasons:
        notes.append(f"stopped by wall-clock limit, compared for t ≤ {t_end:.2f}τ")
    elif any(r["time_matched"] for r in res) or row.evaluations[0].t_range < row.baseline.t_range:
        notes.append(f"compared for t ≤ {t_end:.2f}τ")
    if X.stop_reason(base) == "diverged":
        notes.append("PCM-only also diverged")
    return [r["d_rho"] for r in res], [r["d_v"] for r in res], "; ".join(notes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=X.DEFAULT_MANIFEST)
    p.add_argument("--tables", nargs="*", help="table names (default: all)")
    p.add_argument("--latex", action="store_true", help="LaTeX tabular rows instead of Markdown")
    args = p.parse_args()

    manifest = X.load(args.manifest)
    for table in args.tables or manifest["tables"]:
        print(f"\n{'% ' if args.latex else '### '}{manifest['tables'][table]['title']}\n")
        if not args.latex:
            print("| Row | Δρ | Δv | Note |\n|---|---|---|---|")
        for row in X.table_rows(manifest, table):
            vals = row_values(manifest, row)
            if vals is None:
                cells = ["(missing evaluation)", "", ""]
            else:
                cells = [fmt(vals[0], args.latex), fmt(vals[1], args.latex), vals[2]]
            if args.latex:
                print(f"{row.label} & {cells[0]} & {cells[1]} \\\\" + (f" % {cells[2]}" if cells[2] else ""))
            else:
                print(f"| {row.label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
