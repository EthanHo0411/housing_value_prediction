# Experiments

One thin script per sub-task: it sets `Config` values, calls `housing/`, saves its runs with `save_run()` and draws figures from saved runs. Run each from the repository root:

```bash
python -m experiments.t1a1_learning_rate
```

A1–A5 are implemented; the Task 1B and Task 2 scripts below remain planned.
A1–A5 call `housing/train.py` directly and use `housing/report.py` for JSON/CSV/PNG output.
See the root README for reproduction commands.

| Script | Sub-task | What it does |
|---|---|---|
| `t1a1_learning_rate.py` | 1A1 | learning-rate grid, fold 0 as validation |
| `t1a2_dropout.py` | 1A2 | a few dropout rates, fold 0 as validation |
| `t1a3_cv_dropout.py` | 1A3 | tune the dropout rate with five-fold CV |
| `t1a4_l1_loss.py` | 1A4 | MAE (L1) loss, retune dropout, compare with MSE |
| `t1a5_final_test.py` | 1A5 | train on all 500, test once |
| `t1b1_moment_matching.py` | 1B1 | 1990 network + 2020 statistics, CV, no training |
| `t1b2_finetune.py` | 1B2 | fine-tune vs from scratch, same settings, CV |
| `t2a1_build_check.py` | 2A1 | build the network, check 17,089 parameters |
| `t2a2_overfit.py` | 2A2 | train from scratch, show overfitting |
| `t2a3_dropout.py` | 2A3 | dropout placements compared |
| `t2a4_lora.py` | 2A4 | LoRA rank 4, with and without dropout |
| `t2b_transfer_init.py` | 2B1, 2B2 | MLP part starts from the 1990 weights, then + dropout, LoRA |
| `t2c_variations.py` | 2C | C1.i + C1.ii, or your own C2, with CV |
| `t2d_ablation.py` | 2D | ablation table with CV, then final test once |

Rules that keep comparisons fair: change only `Config` fields between runs you compare, use `fold_split()` / `cross_validate()` for every split, and call `load_test()` only in `t1a5_final_test.py` and `t2d_ablation.py`.
