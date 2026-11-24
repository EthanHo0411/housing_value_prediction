# Task 2C: architectural variations, option C1 (C1.i price encoding, C1.ii layer norm)

| Script | Sub-task | Status |
|---|---|---|
| `housing/attention_extras.py` | C1.i and C1.ii variants (P4) | done (P4) |
| `experiments/t2_extras_check.py` | checks for the variants (P4) | done, all pass |
| `experiments/t2c_variations.py` | five-fold CV of the 2A network, C1.i, C1.ii | done |

## Commands (from the repository root)

| Report item | Command | Output in `results/` | Time |
|---|---|---|---|
| 2C table and figure | `python -m experiments.t2c_variations` | `t2c_variations.json`, `.csv`, `.png` | ~10 min (6 processes) |
| 2C table and figure from saved runs | `python -m experiments.t2c_variations --from-saved` | same files, no training | seconds |

## Protocol

- Variants (P4's `make_model_fn`): the 2A network (17,089 trainable parameters); C1.i, the 1990
  price encoded with 32 frequencies log-spaced from 0.5 to 32, source token 128-dimensional
  (25,153); C1.ii, four separate layer norms on `h2`, `phi(2020)`, `g2`, `phi(1990)` before the
  additions (17,601).
- Five-fold CV with the Task 1A baseline configuration (`configs/task1a_baseline.json`): lr 0.001,
  500 full-batch Adam steps, dropout 0.1 on the MLP hidden units, L1 loss, seed 42 + fold, a
  fresh model and fresh standardisation statistics per fold. CV MAE at the final epoch, mean +/-
  sample s.d. (ddof 1). These are the settings of the Task 1A MLP row, so the cross-attention row
  here is also 2D's "cross-attention" row.
- Same folds and seeds for every variant, so the fold-by-fold difference from the baseline is
  reported as well (it removes the fold-to-fold difficulty).

## Results (five-fold CV, recorded on Windows, CPU, Python 3.14.3, torch 2.14.1)

| Variant | Parameters | CV MAE at epoch 500 | Difference from cross-attention (folds better) | Best mean validation MAE (epoch) |
|---|---|---|---|---|
| Cross-attention (2A network) | 17,089 | 185,078 +/- 15,346 | 0 | 163,169 (80) |
| C1.i price encoding | 25,153 | 206,661 +/- 11,275 | +21,583 +/- 15,244 (1/5) | 163,779 (80) |
| C1.ii layer norm | 17,601 | 190,716 +/- 13,661 | +5,637 +/- 8,650 (2/5) | 166,977 (90) |

- Neither variant improves on the 2A network. Both fit the training folds faster (lower training
  MAE) and overfit more by epoch 500.
- C1.i is worse on 4 of 5 folds (+$21.6k at epoch 500): the 64-dimensional price encoding adds
  8,064 parameters to `W_K` and `W_V`.
- C1.ii is within the fold-to-fold noise of the baseline (+$5.6k +/- $8.7k, better on 2 folds).
- At their best epochs (80-90) the three mean curves are within $3.8k of each other: the
  variations change how fast the network overfits, not how well it fits.
- For 2D: with these settings the cross-attention network (185,078 +/- 15,346) is worse than the
  Task 1A MLP (157,609 +/- 11,044, same configuration), because its validation MAE is lowest
  near epoch 80 and rises by epoch 500. The epoch budget for the 2D rows is still a group decision.
