# Trained correction models of the report

Final checkpoints of every SuperadiabaticNet (or PowerFunctionalNet) model behind the
report results, all trained with the fixed pipeline (100 solver substeps per MD frame,
force_alpha = 1). Each `<name>.ckpt` was trained with the config listed for `<name>` in
[`experiments/report.yaml`](../../experiments/report.yaml), which also lists the seed. The
configs are in `configs/report/`. All models use the frozen PCM functional
`models/c2_dz32.ckpt`.

| Folder | Models | Report |
|---|---|---|
| `density_weighted/` | 3a (seeds 42, 43, 44), gaussian (42, 43), 3c, mg1, mg2, s1 | Tables 1 and 3, sources for the zero-shot and friction transfer |
| `loss_weight/` | velocity-weighted (w_ρ/w_v = 1/10) on 3a, gaussian, mg1; gaussian with w_ρ = 18.7; mg1 with w_ρ = 39.4 | Table 2, Sec. 6.3 |
| `inputs/` | 3a with other input channels: minimal, + acceleration, + history and V_ext (seeds 42, 43), + history, acceleration and V_ext | Table 5 (the full input set is `density_weighted/3a`) |
| `friction/` | gaussian trained at γ = 0.05 | Table 4 |
| `multi_potential/` | 3a + gaussian (32 epochs), 3a + gaussian + mg1 + s1 (stopped after 70 of 80 epochs) | Sec. 6.6, Table 6 |
| `alternative_outputs/` | scalar-potential output and PowerFunctionalNet, 10-epoch smoke tests | Discussion |

Evaluate a model with its own training config, which holds the input channels and
physics parameters:

```bash
python scripts/evaluation/phiflow_eval.py --config configs/report/density_weighted/3a.yaml \
    --model_mode hybrid --sup_model_path models/report/density_weighted/3a.ckpt \
    --md_path ../data/lj_data/ref_3a_langevin_01/avg_2000 --out_dir results/my_eval
```

`scripts/reproduce/evaluate.py` does this for all report evaluations.
