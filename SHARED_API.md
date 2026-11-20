# Predicting housing values in 2020

70050 Machine Learning, Coursework 2. Learn 2020 California house values from 500 examples, then test whether the 1990 data helps: by transfer from a network trained on 1990 (Task 1B) and by cross-attention to the 20,640 records from 1990 (Task 2).

This repository holds the **shared code** every experiment uses (`housing/`), the self-checks (`checks.py`), the data, and one script per sub-task in `experiments/`.

## Setup

On a lab machine, use the lab environment the TAs run the code in:

```bash
source /vol/lab/ml/mlvenv/bin/activate
```

On your own machine, mirror its versions (Python 3.14, see `requirements.txt`):

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Then, from the repository root:

```bash
python checks.py
```

All checks must pass before you run experiments. They take about 20 s.

## Layout

```
README.md                  this file: setup, conventions, and the command behind every figure and table
requirements.txt           library versions of the lab environment /vol/lab/ml/mlvenv
checks.py                  [SHARED]  self-checks of the shared code
data/                      [SHARED]  the supplementary files, unchanged
  housing_2020_train.csv             500 rows + fold column            -> all tasks
  housing_2020_test.csv              21,282 rows                       -> 1A5 and 2D only (final test)
  housing_1990.csv                   20,640 rows                       -> Task 2 (the 1990 records)
  mlp_1990.pt, mlp_1990.json         1990 network and its statistics   -> 1B, 2A-2D
housing/                   shared library, imported by every experiment
  data.py                  [SHARED]  load the CSVs, fixed column order, fold_split(k), load_test()
  scaling.py               [SHARED]  Standardiser: fit on training rows, standardise, back to dollars
  mlp.py                   [SHARED]  MLP(drop_rate), load_mlp_1990(); also used inside Task 2
  train.py                 [SHARED]  Config, train_full_batch(), cross_validate(), predict_dollars()
  report.py                [SHARED]  save_run()/load_run(), plot_curves(), summary_table()
  attention.py             [TASK 2]  to be written: position encoding, cross-attention, LoRA, layer norm
experiments/               one script per sub-task, see experiments/README.md
results/                   [OUTPUT]  configs and curves (JSON) and figures written by the scripts (git-ignored)
```

## How an experiment uses the library

An experiment script sets a `Config`, calls the shared code, saves the run, and draws from saved runs. Run scripts as modules from the repository root, so `housing` imports work: `python -m experiments.<script_name>`.

```python
from housing.data import fold_split
from housing.report import load_run, plot_curves, save_run
from housing.train import Config, cross_validate, train_full_batch

# one run: train on folds 1-4, validate on fold 0
cfg = Config(lr=1e-3, epochs=2000, drop_rate=0.0, loss="mse", seed=0)
fit = train_full_batch(cfg, *fold_split(0))
save_run("example_fold0", cfg, [fit.history])

# five-fold cross-validation of the same setting
cv = cross_validate(cfg)
save_run("example_cv", cfg, cv.histories)
mean, sd = cv.summary()            # CV MAE in dollars at the last epoch

# figures come from saved runs, not from retraining
plot_curves({"lr 1e-3": load_run("example_cv")}, "results/example_cv.png")
```

Two more patterns the shared code covers:
- **Moment matching** (1B1) is `cross_validate(Config(lr=0.0, epochs=0, init="1990"))`: the 1990 network, 2020 statistics from each round's training folds, no training.
- **Fine-tuning** (1B2) is `Config(..., init="1990")`; its epoch-0 validation MAE equals moment matching, fold by fold (checked in `checks.py`).

## Conventions (agreed defaults; change them in one place if the group decides otherwise)

| Convention | Choice | Where |
|---|---|---|
| Standard deviation | population (`ddof=0`), as in `mlp_1990.json` | `housing/scaling.py`, `DDOF` |
| Seed | the same seed in every fold, set before the model is built | `Config.seed` |
| Training MAE | measured with dropout switched off, like validation MAE | `housing/train.py`, `record()` |
| Curves | entry 0 is before the first step; one epoch is one full-batch step | `History` |
| `drop_rate` | probability of **dropping** a unit (PyTorch's convention; the Week 4 slides use p for keeping one) | `housing/mlp.py` |
| Model interface | every model is called as `model(x_std, x_raw)`; the MLP ignores `x_raw` | `housing/mlp.py` |
| Test set | only through `load_test()`, only for 1A5 and 2D, once each | `housing/data.py` |
| Results | `results/` is git-ignored; regenerate with the commands below | `.gitignore` |

## Reproducing the report

Every figure and table in the report, and the command that produces it (fill in as experiments are added):

| Report item | Command |
|---|---|
| Self-checks | `python checks.py` |
| 1B1 moment-matching table | `python -m experiments.t1b1_moment_matching` |
| 1B2 fine-tune vs scratch figure and table | `python -m experiments.t1b2_finetune` |
| 1B2 longer budget (2000 epochs) | `python -m experiments.t1b2_finetune --long` |
| 1B2 settings grid | `python -m experiments.t1b2_finetune --sweep` |
