# Legacy

Scripts and modules superseded by the PhiFlow-based pipeline (`scripts/training/train_phiflow.py`,
`scripts/evaluation/phiflow_eval.py`, `src/neural_ddft/physics/phiflow_pde.py`), kept for reference only:

- `scripts/evaluation/{ddft_fmt,ddft_neural_closed*,pypde_eval}.py`, `scripts/training/train_ddft_phiflow.py`,
  `scripts/tools/ddft_debug.py` — depend on the old `py-pde` library, replaced because `py-pde` was unstable
  for friction coefficients (`gamma`) below 2 (see `DEV_STATE.md`).
- `scripts/training/train_ddft_pft_hybrid.py` — an earlier PhiFlow-based trainer predating the unified
  curriculum trainer in `train_phiflow.py` (no curriculum/config support).
- `src/neural_ddft/physics/fmt/` — classical Fundamental Measure Theory free-energy functional, used only
  by the legacy `ddft_fmt.py`.
- `_functions_copy.py` — a stale duplicate of `src/neural_ddft/utils/functions.py`.

None of these are imported by the active pipeline. Not covered by tests.
