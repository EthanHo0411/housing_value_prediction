# Predicting housing values in 2020

70050 Machine Learning, Coursework 2. We predict California house values in 2020 from 500
training examples, and test whether 20,640 records from 1990 help: by transferring a network
trained on 1990 data (Task 1B) and by cross-attention to the 1990 records (Task 2).

## 1. Setup

On a lab machine, use the lab environment (the one the code is checked with):

```bash
source /vol/lab/ml/mlvenv/bin/activate
```

Use the preinstalled Lab packages without installing or upgrading them. On another
machine, match the actual Lab Python and package versions in an isolated environment.
Historical versions are recorded in
[short_summary/ENVIRONMENT.md](short_summary/ENVIRONMENT.md).

Run every command below **from the repository root**. Experiments are run as modules:
`python -m experiments.<script>`.

## 2. Check that everything works

```bash
python checks.py                          # shared code: data, folds, standardisation, training (~20 s)
python checks_task1a.py                   # Task 1A integration checks
python -m experiments.t2a1_build_check    # Task 2 network: 17,089 parameters, shapes, seeds (~10 s)
python -m experiments.t2_extras_check     # dropout, LoRA, price encoding, layer norm (~30 s)
```

None of these read the test set or write results.

## 3. Reproduce the results

All results are already saved in `results/`. To **redraw a figure or table without training**,
add `--from-saved` where the table says so (seconds). To **retrain**, run the command as shown (see the Task 1A5 note below).
Run experiments in table order; for example, A4 needs the A3 results.
Times are approximate, on a laptop CPU.

| Task | Command | Writes to `results/` | Time |
|---|---|---|---|
| 1A1 learning rate | `python -m experiments.t1a1_learning_rate` | `t1a1_learning_rate.*` | < 1 min |
| 1A2 dropout | `python -m experiments.t1a2_dropout --learning-rate 0.001` | `t1a2_dropout.*` | < 1 min |
| 1A3 tune dropout (CV) | `python -m experiments.t1a3_cv_dropout --learning-rate 0.001 --epochs 500` | `t1a3_cv_dropout.*` | < 1 min |
| 1A4 MAE vs MSE (CV) | `python -m experiments.t1a4_l1_loss` | `t1a4_l1_loss.*`, `t1a4_loss_comparison.*` | < 1 min |
| 1A5 final test | `python -m experiments.t1a5_final_test` | `t1a5_final_test.json`, `t1a5_model.pt` | < 1 min |
| 1B1 moment matching | `python -m experiments.t1b1_moment_matching` | `t1b1_moment_matching*.json` | ~5 s |
| 1B2 fine-tuning | `python -m experiments.t1b2_finetune` (`--long`: 2000 epochs; `--sweep`: settings grid) | `t1b2_finetune*` | ~1 min (sweep ~4 min) |
| 2A1 build check | `python -m experiments.t2a1_build_check` | nothing | ~10 s |
| 2A2 overfitting | `python -m experiments.t2a2_overfit` (or `--from-saved`) | `t2a2_overfit*` | ~10 min |
| 2A3 dropout | `python -m experiments.t2a3_dropout` (or `--from-saved`) | `t2a3_dropout*` | ~20 min |
| 2A3 diagnostics | `python -m experiments.t2a3_diagnostics` | `t2a3_diagnostics.json` | ~15 min |
| 2A4 LoRA | `python -m experiments.t2a4_lora` (or `--from-saved`) | `t2a4_lora*` | ~20 min |
| 2B1 transferred init | `python -m experiments.t2b_transfer_init` (or `--from-saved`) | `t2b1_transfer_init*` | ~10 min |
| 2B2 + dropout, LoRA | `python -m experiments.t2b_transfer_init --b2` (or `--b2 --from-saved`) | `t2b2_dropout_lora*` | ~30 min |
| 2C architectural variations (CV) | `python -m experiments.t2c_variations` (or `--from-saved`) | `t2c_variations*` | ~20 min |
| 2D ablation (CV) | `python -m experiments.t2d_ablation` (or `--from-saved`) | `t2d_ablation*` | ~40 min |
| 2D final test | `python -m experiments.t2d_ablation --final-test "cross-attention + LoRA + weight dropout"` | `t2d_final_test.json`, `t2d_model.pt` | ~5 min |

**Task 1A5:** The command displays the saved MAE if a result already exists.
To retrain, use a separate repository copy. In that copy, remove
`results/t1a5_final_test.json` and `results/t1a5_locked_config.json`, keep the
A3/A4 results, and run the A5 command once. Do not tune settings on the test set.

Task 1A figures and tables can also be redrawn from saved results, e.g.
`python -m housing.report results/t1a1_learning_rate.json`.

More detail on each task (protocol, settings, full result tables) is in [TASK_1B.md](short_summary/TASK_1B.md),
[TASK_2A.md](short_summary/TASK_2A.md), [TASK_2C.md](short_summary/TASK_2C.md) and [TASK_2D.md](short_summary/TASK_2D.md).

## 4. Good to know

- **Test set.** Only the two final tests read `data/housing_2020_test.csv`: 1A5 and the 2D
  final test. Model choices were fixed by cross-validation before the test set was read. 1A5
  refuses to rerun once its result exists; run the 2D final test only once.
- **Memory.** The Task 2 scripts train several runs in parallel; each worker needs about 0.5 GB.
  On a laptop with 8 GB of RAM, add `--processes 2` (to `t2a2_overfit`, `t2a3_dropout`,
  `t2a4_lora`, `t2b_transfer_init`, `t2c_variations`), or the runs can stall.
- **Reproducibility.** Fixed seeds help repeat runs in the same environment.
  Previous cross-machine comparisons differed by about \$100 after 500 epochs and
  up to about \$8k after 2000 epochs, without changing those runs' conclusions.
  These are observations, not guaranteed limits. Actual Lab execution still needs verification.
- **Rerunning overwrites results.** An experiment rewrites its files in `results/`. Use
  `git diff results/` to compare with the saved ones, or `git checkout results/` to restore them.

## 5. Repository layout

```
housing/          shared library, used by every experiment
  data.py           load the CSVs, folds, test set
  scaling.py        standardisation (fitted on training rows only)
  train.py          full-batch training, cross-validation, MAE in dollars
  mlp.py            the Task 1 MLP and the provided 1990 network
  attention.py      the Task 2 cross-attention network
  attention_extras.py  dropout options, LoRA, price encoding, layer norm
  report.py         saving results and drawing figures
experiments/      one script per sub-task (commands above)
data/             the supplied data and the 1990 network, unchanged
results/          saved results (JSON), figures and models
configs/          the Task 1A baseline configuration
short_summary/    shared API, task notes and historical environment versions
checks.py, checks_task1a.py   self-checks
```
