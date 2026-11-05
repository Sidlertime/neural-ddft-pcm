import os
import json
import numpy as np


def write_run_summary(results_path, config, force_ratios, blowup_info,
                       n_steps_completed, final_sim_time, wall_clock_seconds,
                       reference_comparison):
    """
    Writes a single fast-to-read summary.json for an evaluation run, so a run's
    outcome (did it blow up, how big was the superadiabatic correction, how
    close did it land to the MD reference) can be assessed without opening the
    rendered movie or re-running compare_profiles.py by hand.
    """
    summary = {
        "run_id": config.get("run_id"),
        "model_mode": config.get("model_mode"),
        "config_echo": config,
        "n_steps_completed": n_steps_completed,
        "final_sim_time": final_sim_time,
        "blowup": blowup_info,
        "force_ratio": {
            "mean_over_trajectory": float(np.mean(force_ratios)) if force_ratios else None,
            "max_over_trajectory": float(np.max(force_ratios)) if force_ratios else None,
            "final": force_ratios[-1] if force_ratios else None,
        },
        "reference_comparison": reference_comparison,
        "wall_clock_seconds": wall_clock_seconds,
    }
    with open(os.path.join(results_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
