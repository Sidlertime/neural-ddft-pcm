"""
Experiment manifest (experiments/report.yaml): which models, evaluations and
tables make up the report, where their outputs live, and how to run them.

Output layout (below RESULTS_ROOT/report/):
    train/<model>/                       training runs (scripts/reproduce/train.py)
    eval/<model>/<target><suffix>/       model evaluations
    eval/baseline/<target><suffix>/      PCM-only (equilibrium) evaluations
where <suffix> encodes non-default conditions, e.g. "_gamma0.05", "_dt5e-05".
"""
import glob
import os
import sys
from dataclasses import dataclass, field

import yaml

from neural_ddft.utils import paths

DEFAULT_MANIFEST = os.path.join(paths.DDFT_ROOT, "experiments", "report.yaml")
SHIPPED_MODELS = os.path.join(paths.DDFT_ROOT, "models", "report")


def load(manifest=None):
    with open(manifest or DEFAULT_MANIFEST) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Evaluation:
    """One evaluation run: a model (or the PCM-only baseline if model is None) on a target."""
    model: str | None
    target: str
    gamma: float
    dt: float
    t_range: float
    diffusion_coeff: float | None = None

    @property
    def is_baseline(self):
        return self.model is None

    def suffix(self, defaults):
        s = ""
        if self.gamma != defaults["gamma"]:
            s += f"_gamma{self.gamma:g}"
        if self.dt != defaults["dt"]:
            s += f"_dt{self.dt:g}"
        if self.t_range != defaults["t_range"]:
            s += f"_t{self.t_range:g}"
        if self.diffusion_coeff is not None:
            s += f"_diff{self.diffusion_coeff:g}"
        return s

    def out_dir(self, defaults):
        who = "baseline" if self.is_baseline else self.model
        return os.path.join(paths.RESULTS_ROOT, "report", "eval", who, self.target + self.suffix(defaults))

    def name(self, defaults):
        return f"{'baseline' if self.is_baseline else self.model} -> {self.target}{self.suffix(defaults)}"


@dataclass
class Row:
    label: str
    models: list
    target: str
    evaluations: list = field(default_factory=list)   # one per model (seed)
    baseline: Evaluation = None


def table_rows(manifest, table):
    d = manifest["defaults"]
    rows = []
    for r in manifest["tables"][table]["rows"]:
        models = r.get("models") or [r["model"]]
        gamma, dt, t_range = r.get("gamma", d["gamma"]), r.get("dt", d["dt"]), r.get("t_range", d["t_range"])
        row = Row(label=r.get("label", f"{models[0]} -> {r['target']}"), models=models, target=r["target"])
        row.evaluations = [Evaluation(m, r["target"], gamma, dt, t_range) for m in models]
        # the baseline always runs the full time; shorter model runs are compared time-matched
        row.baseline = Evaluation(None, r["target"], gamma, d["dt"], d["t_range"])
        rows.append(row)
    return rows


def extra_evaluations(manifest):
    """Evaluations needed only for figures (listed under `figures_extra` in the manifest)."""
    d = manifest["defaults"]
    out = []
    for e in manifest.get("figures_extra", []):
        out.append(Evaluation(e.get("model"), e["target"], e.get("gamma", d["gamma"]), e.get("dt", d["dt"]),
                              e.get("t_range", d["t_range"]), e.get("diffusion_coeff")))
    return out


def all_evaluations(manifest, tables=None):
    """Unique evaluations (models and baselines) of the given tables (default: all) plus figure extras."""
    seen = {}
    for t in tables or manifest["tables"]:
        for row in table_rows(manifest, t):
            for e in row.evaluations + [row.baseline]:
                seen.setdefault(e, None)
    if tables is None:
        for e in extra_evaluations(manifest):
            seen.setdefault(e, None)
    return list(seen)


def model_config(manifest, model):
    return os.path.join(paths.DDFT_ROOT, manifest["models"][model]["config"])


def model_mode(manifest, model):
    with open(model_config(manifest, model)) as f:
        return yaml.safe_load(f).get("model_mode", "hybrid")


def trained_checkpoint(model):
    """Last checkpoint of a run trained with scripts/reproduce/train.py, or None."""
    ckpts = sorted(glob.glob(os.path.join(paths.RESULTS_ROOT, "report", "train", model, "checkpoints", "*.ckpt")))
    return ckpts[-1] if ckpts else None


def checkpoint(model, source="shipped"):
    if source == "trained":
        ckpt = trained_checkpoint(model)
        if ckpt is None:
            raise FileNotFoundError(f"no trained checkpoint for '{model}'; run scripts/reproduce/train.py first")
        return ckpt
    return os.path.join(SHIPPED_MODELS, model + ".ckpt")


def eval_command(manifest, e, source="shipped", max_wall_clock=4 * 3600):
    """Command line of scripts/evaluation/phiflow_eval.py for one evaluation (run in DDFT_ROOT)."""
    d = manifest["defaults"]
    cmd = [sys.executable, "scripts/evaluation/phiflow_eval.py"]
    if e.is_baseline:
        cmd += ["--model_mode", "equilibrium"]
    else:
        mode = model_mode(manifest, e.model)
        ckpt_flag = "--power_model_path" if mode == "hybrid_power" else "--sup_model_path"
        cmd += ["--config", model_config(manifest, e.model), "--model_mode", mode,
                ckpt_flag, checkpoint(e.model, source)]
    cmd += ["--base_model_path", d["base_model_path"],
            "--md_path", paths.md_dir(e.target, e.gamma),
            "--gamma", str(e.gamma), "--eta", str(d["eta"]), "--dt", str(e.dt), "--t_range", str(e.t_range),
            "--max_wall_clock_seconds", str(max_wall_clock),
            "--run_id", e.name(d).replace(" -> ", "_on_").replace("/", "_"),
            "--out_dir", e.out_dir(d)]
    if e.diffusion_coeff is not None:
        cmd += ["--diffusion_coeff", str(e.diffusion_coeff)]
    return cmd


def train_command(manifest, model):
    """Command line of scripts/training/train_phiflow.py for one model (run in DDFT_ROOT)."""
    spec = manifest["models"][model]
    cmd = [sys.executable, "scripts/training/train_phiflow.py", "--config", model_config(manifest, model),
           "--out_dir", os.path.join(paths.RESULTS_ROOT, "report", "train", model)]
    if "seed" in spec:
        cmd += ["--seed", str(spec["seed"])]
    return cmd


def is_done(e, manifest):
    return os.path.exists(os.path.join(e.out_dir(manifest["defaults"]), "summary.json"))


def n_saved(out_dir):
    """Number of saved density frames of an evaluation folder."""
    return len(glob.glob(os.path.join(out_dir, "rho*.npy")))


def stop_reason(out_dir):
    """None if the evaluation reached its end time, else "diverged" or "time limit" (from summary.json)."""
    import json
    with open(os.path.join(out_dir, "summary.json")) as f:
        blowup = json.load(f).get("blowup") or {}
    if not blowup.get("occurred"):
        return None
    return "time limit" if "wall-clock" in (blowup.get("reason") or "") else "diverged"
