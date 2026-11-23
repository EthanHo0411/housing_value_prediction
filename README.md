# Coursework 2 — Task 1A1–1A5

This implementation follows Section 2 (pp. 3–4) and Task 1A (p. 5) of
`70050_5_spec.pdf`, using the shared-library and thin-experiment-script structure
in `Week6_coursework2_shared_code_layout.pdf`.
Task 1B experiments, Task 2 experiments, and `attention.py` are outside this implementation's scope.
The original supplementary files are copied unchanged into `data/`.

## Structure and specification mapping

| File | Purpose |
|---|---|
| `housing/data.py` | Fixed order of six features, supplied folds, 400/100 splits, and final test loading |
| `housing/scaling.py` | Training-only means and standard deviations, input/price standardisation, and conversion back to dollars |
| `housing/mlp.py` | 6→64→64→1 ReLU MLP, dropout in both hidden layers, and a 1990 model loader for later Task 1B work |
| `housing/train.py` | PyTorch default initialisation, full-batch Adam, MSE/L1, five-fold CV, and dollar MAE |
| `housing/report.py` | PNG curves and CSV tables generated from saved JSON results |
| `checks.py` | Checks for leakage, splits, reproducibility, model structure, and evaluation calculations |
| `experiments/t1a1_learning_rate.py` | 1A1: learning-rate comparison on fold 0 for 2000 epochs |
| `experiments/t1a2_dropout.py` | 1A2: change only dropout, using fold 0 for 2000 epochs |
| `experiments/t1a3_cv_dropout.py` | 1A3: tune dropout for MSE using the supplied five folds |
| `experiments/t1a4_l1_loss.py` | 1A4: retune dropout for L1 under the same settings as A3 and compare losses |
| `experiments/t1a5_final_test.py` | 1A5: fix settings using CV, train on all 500 rows, and evaluate on the test set |

## Environment

Locally verified with Python 3.12 on CPU. The verified direct dependency versions
are pinned in `requirements-local.txt`; runtime details and data SHA-256 hashes are in
`results/environment.json`.
**The Lab environment at `/vol/lab/ml/mlvenv` is not accessible from this machine, so version compatibility has not been verified.**
`requirements.txt` preserves the existing shared repository dependency pins.
`requirements-local.txt` records the locally verified versions, including pandas required by the shared loader.
On a Lab machine, run `checks.py` using the Lab Python and compare dependency versions as needed.

To create a new environment, run the following from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

To use the existing verification environment on this machine (the virtual environment remains at its original location):

```bash
cd "/Users/ruka/housing_value_prediction"
source "/Users/ruka/Downloads/data 2/.venv312/bin/activate"
export MPLCONFIGDIR="$PWD/.mplconfig"
```

## Running experiments and generating outputs

Run all commands below from the repository root, starting with `python checks.py`.
Use `python -m ...` to run experiment modules.
Inspect the A1 and A2 curves before choosing the learning rate and budget for A3.
A4 inherits the settings and dropout candidates from the current A3 results.
A5 also selects the loss function by comparing the best CV MAE from A3 and A4.

```bash
python checks.py
python -m experiments.t1a1_learning_rate
python -m experiments.t1a2_dropout --learning-rate 0.001
python -m experiments.t1a3_cv_dropout --learning-rate 0.001 --epochs 500
python -m experiments.t1a4_l1_loss
python -m experiments.t1a5_final_test
```

Each experiment saves its settings, per-epoch curves, and standardisation statistics
in `results/<script_name>.json`. Matching `.png` files contain curves and `.csv` files
contain result tables. The A3/A4 CSV files include each fold's MAE, the mean, and the
sample standard deviation (ddof=1). A4 also saves tables and curves comparing all
MSE and L1 candidates in `t1a4_loss_comparison.*`. A5 saves JSON, CSV, and a model checkpoint.

Regenerate figures and tables from saved results without retraining or accessing the test set:

```bash
python -m housing.report results/t1a1_learning_rate.json
python -m housing.report results/t1a2_dropout.json
python -m housing.report results/t1a3_cv_dropout.json
python -m housing.report results/t1a4_l1_loss.json
python -m housing.report results/t1a4_loss_comparison.json
python -m housing.report results/t1a5_final_test.json
```

## Experimental design and evaluation definitions

- Input order: `median_income, population, households, housing_age, latitude, longitude`.
  Neither `fold` nor the target is included as an input.
- Standardisation uses population standard deviations (ddof=0), computed from the
  400 training rows in each CV round or all 500 rows for final training. A zero
  standard deviation raises an error in the shared scaler. The statistics are saved with the model.
- Apart from the learning rate, Adam uses PyTorch defaults: betas=(0.9, 0.999),
  eps=1e-8, and weight_decay=0. No scheduler, gradient clipping, or early stopping is used.
- Each epoch is one update using all training rows. Epoch 0 is the initial model;
  epoch t is measured after t updates. `standardised_loss` is the training loss
  computed with dropout immediately before the update, and is distinct from dollar MAE.
- Both training and validation MAE are measured with `eval()` and `no_grad()`, with
  dropout disabled. Evaluation does not consume dropout randomness and restores
  the previous model mode afterwards.
- Runs use CPU, one thread, and deterministic algorithms. The base seed is 42;
  CV fold k uses 42+k. Candidates share initial weights within each fold, and a
  fresh model and optimiser are created for every fold.
- A1 learning-rate candidates: 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0.
  If a run produces non-finite values, its stopping reason is recorded instead of
  writing non-finite values to JSON. A1 plots use a logarithmic vertical axis.
- Dropout candidates: 0, 0.1, 0.2, 0.3, 0.5. The same probability is applied after
  ReLU in both hidden layers.
- CV selection uses the **mean validation MAE across five folds at the fixed final epoch**,
  rather than selecting the best epoch separately for each fold. Ties are resolved
  by candidate order. Shaded bands show ±1 sample standard deviation across folds,
  not confidence intervals.
- Test MAE is the sum of absolute errors across all batches divided by 21,282.
  The smaller final batch is weighted by its sample count, rather than receiving
  the same weight as a full batch.

## Preventing repeated final test evaluation

A5 fixes the configuration by exclusively creating `t1a5_locked_config.json`, then
trains on all 500 rows before reading the test CSV for the first time. Model weights,
statistics, feature order, and settings are saved in `t1a5_model.pt`.
If the result JSON already exists, rerunning A5 only displays the saved result;
there is no further test access.
If only the lock file remains, execution stops. Before recovering an interrupted run,
check the cause and whether test evaluation already took place. Do not retune settings
based on the test results.
A complete reproduction can be run in a fresh copy without results or a lock file,
using the same fixed settings.

Restore the model without reevaluating the test set:

```python
from housing.mlp import load_checkpoint
from housing.train import predict_dollars

model, scaler, config = load_checkpoint("results/t1a5_model.pt")
# Accepts both the original Sequential checkpoint and the unified MLP format.
# To predict new data, wrap raw inputs and placeholder targets in Rows,
# then call predict_dollars(model, scaler, rows).
```

## Points to cover in the report

For the recorded runs, A3 onwards uses learning rate 0.001 and 500 epochs.
In A1, 0.001 gave rapid initial improvement without the large initial spikes observed
at 0.1 and 1.0. In A2, dropout rates 0.1, 0.2, and 0.5 reached their lowest or near-lowest
fold 0 validation MAE around epochs 300–550; extending training to 2000 epochs did not
improve on those values. A common budget of 500 epochs was therefore selected before
running CV. This budget choice is based on exploration with fold 0, rather than
using CV itself to tune the epoch count or selecting settings from test results.

- 1A1: Use the curves to identify slow progress at low learning rates and instability at high rates.
- 1A2: Compare the training/validation gap and how it changes with dropout.
- 1A3: Explain the learning rate and budget choices; report the selected dropout and CV MAE mean ± standard deviation.
- 1A4: Retune dropout for each loss, then compare MSE and L1 on the same folds.
- 1A5: Report the fixed settings and test MAE.

Use the saved results as evidence, and verify and write your own analysis and conclusions for submission.

## Recorded results

The following values are from the completed runs. All values are in dollars;
CV results are the five-fold mean ± sample standard deviation.

| Dropout | 1A3: MSE-trained CV MAE | 1A4: L1-trained CV MAE |
|---:|---:|---:|
| 0.0 | 189,373.72 ± 10,380.53 | 165,480.80 ± 10,017.77 |
| 0.1 | 165,685.26 ± 5,504.68 | 157,609.08 ± 11,043.99 |
| 0.2 | 165,826.82 ± 10,691.29 | 159,338.76 ± 11,362.15 |
| 0.3 | 167,761.75 ± 10,884.12 | 161,719.01 ± 12,439.57 |
| 0.5 | 166,955.74 ± 12,683.85 | 161,338.15 ± 15,288.41 |

A5 uses **L1, dropout=0.1, learning rate=0.001, 500 epochs, and seed=42**.
After training on all 500 examples, the MAE on 21,282 test examples was **$161,795.67**.
The test set was evaluated once, after these settings had been fixed.

The table corresponds to `t1a3_cv_dropout.csv` and `t1a4_l1_loss.csv`; the final
result is in `t1a5_final_test.csv`. Even with fixed seeds, exact numerical agreement
is not guaranteed across different PyTorch versions or hardware.

## Unified repository and shared API

There is one `housing/`, `experiments/`, `data/`, and `results/` at the repository root.
The former nested `cw2/` has been merged into this structure.
The shared `Rows`, `MLP`, `Config(lr=..., drop_rate=...)`, `Fit`, and `CVResult` APIs
are preserved for Tasks 1B and 2. See [SHARED_API.md](SHARED_API.md) for usage.
Each experiment calls `train_full_batch()` or `cross_validate()` in `housing/train.py` directly.
`housing/report.py` only serialises completed results and renders figures; it never trains models.
Checkpoint loading lives in `housing/mlp.py`, and scaler serialisation lives in `housing/scaling.py`.
New results use the shared Config field names `lr` and `drop_rate`.
Historical result files remain readable through `Config.from_dict()`.

The shared CV defaults remain a constant seed across folds and ddof=0 for summary
standard deviations. Task 1A explicitly uses `fold_seed_stride=1` and `summary(ddof=1)`
to preserve its original seed and reporting conventions. Input standardisation uses
ddof=0 in both cases. Existing results and the final-test lock are preserved unchanged;
merging does not trigger another test evaluation.

Verify the integration without test access or result writes:

```bash
python checks.py
python checks_task1a.py
python checks_task1a.py --reproduce
```

The optional reproduction check reruns A1–A4 using the unified implementation and
compares against the recorded results. It requires the locally verified dependency versions.

Integration verification: all original shared checks passed, and all A1–A4 candidates
reproduced the recorded MAE curves/fold scores (maximum observed validation difference
$0.000000 at six decimal places in the local environment). Both checkpoint formats
and the A5 execution path were checked using training data as a stand-in; the real
test set was not reevaluated.

## Shared baseline and tracked reference results

`configs/task1a_baseline.json` is the fixed Task 1A baseline for other group members:
L1 loss, learning rate 0.001, dropout 0.1, 500 epochs, seed 42, and default initialisation.
It records the CV seed stride, standard deviation convention, preprocessing, and runtime
settings separately from the model training configuration. It was selected using CV,
not the test score. When comparing later tasks, state any intentional changes to it.

Load the baseline from the repository root:

```python
import json
from pathlib import Path
from housing.train import Config, configure_runtime, cross_validate

baseline = json.loads(Path("configs/task1a_baseline.json").read_text())
cfg = Config(**baseline["config"])
configure_runtime()
cv = cross_validate(cfg, fold_seed_stride=baseline["cross_validation"]["fold_seed_stride"])
mean, sd = cv.summary(ddof=baseline["cross_validation"]["summary_ddof"])
```

The explicit Git allowlist includes the A1–A4 result JSON files (including the combined
loss comparison), A5 result and lock JSON files, `environment.json`, and the small final
model checkpoint. These support `checks_task1a.py --reproduce`, A4's input from A3,
figure regeneration, and checkpoint checks on a fresh clone. The A5 result/lock are
included so a fresh clone retains the completed final-evaluation state.
PNG and CSV outputs remain ignored because they can be regenerated from the JSON.
New experiment outputs are ignored unless explicitly added to the allowlist.

Treat these tracked files as reference results. Rerunning experiments may modify them;
review any changes before committing. To verify reproduction without replacing the
reference outputs, run `python checks_task1a.py --reproduce`.
