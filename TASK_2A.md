# Task 2A: cross-attention (P3's part)

| Script | Sub-task | Status |
|---|---|---|
| `housing/attention.py` | 2A1 network | done |
| `experiments/t2a1_build_check.py` | 2A1 checks (~10 s; no test data, writes nothing) | done, all pass |
| `experiments/t2a2_overfit.py` | 2A2 overfitting, learning rates 0.0001-0.003 | done |
| `experiments/t2a4_lora.py` | 2A4 LoRA | waiting for P4's `apply_lora(model, r=4)` (see `P4_LOOKOUT.md`) |

This branch only adds files, so it merges without conflicts. Two things are left for whoever
consolidates the shared files: add the commands below to the root `README.md`, and add
`results/t2a2_overfit.json` and `results/t2a2_overfit_overlay.png` to the `.gitignore`
allow-list (they are tracked here with `git add -f`, so they are shared already).

## Commands (from the repository root, after `python checks.py`)

| Report item | Command | Output in `results/` | Time |
|---|---|---|---|
| 2A1 parameter count and checks | `python -m experiments.t2a1_build_check` | none | ~10 s |
| 2A2 curves and table | `python -m experiments.t2a2_overfit` | `t2a2_overfit.json`, `.png`, `.csv`, `t2a2_overfit_overlay.png` | ~10 min (4 processes) |
| 2A2 summary and figures from saved runs | `python -m experiments.t2a2_overfit --from-saved` | same files, no training | seconds |

## Protocol

- Network: spec section 4, Figure 1; `d_k` = 64; 17,089 trainable parameters (4,673 MLP + 12,416 for `W_Q`, `W_K`, `W_V`, no biases).
- 1990 side, fixed: `g2` from `mlp_1990.pt` (frozen) on 1990-standardised inputs (`mlp_1990.json`); Fourier position encoding of latitude/longitude in km, 16 wavelengths 2-1,000 km; price scaled to [0, 1] with the min/max of `housing_1990.csv` ($14,999 / $500,001).
- Training: Task 1A conventions (`configs/task1a_baseline.json`): L1 loss, seed 42, `configure_runtime()`; full-batch Adam; 2000 epochs; folds 1-4 train, fold 0 validation; no dropout; MAE recorded every 10 epochs.
- Seeds: the MLP part starts identical to Task 1A's MLP for the same seed; `init="1990"` changes only the MLP weights, and leaves the attention weights and the random state (so the dropout masks) identical. Both are checked in `t2a1_build_check.py`.

## Results (fold 0, recorded on Windows, CPU, Python 3.14.3, torch 2.14.1: the lab versions in `requirements.txt`)

| Learning rate | Best validation MAE (epoch) | Final validation MAE | Final train MAE | Mean train MAE, last 10 records |
|---|---|---|---|---|
| 0.0001 | 150,577 (790) | 163,389 | 69,764 | 70,334 |
| 0.0003 | 149,787 (270) | 176,170 | 32,897 | 35,275 |
| 0.001 | 147,323 (120) | 200,922 | 12,279 | 7,933 |
| 0.003 | 145,361 (50) | 197,488 | 4,913 | 6,300 |

At 0.001 and 0.003 the training MAE is noisy below about $15k (its lowest points: $5,548 at epoch 1980 and $3,272 at epoch 1570), so single final readings there are rough; the mean of the last 10 records is steadier.

Same settings for the Task 1A MLP on fold 0 (lr 0.001, L1, no dropout, seed 42, from `results/t1a4_l1_loss.json`, recorded on another machine): best validation 144,914 at epoch 120; at epoch 500 validation 155,318, training 91,768 (attention: 187,143 and 31,738).

## Base setting for the rest of Task 2 (group decision)

Proposed: **learning rate 0.001**, the rate of Task 1A and 1B, so every 2D ablation row can share it down to the Task 1A MLP row. On fold 0 its validation MAE is lowest at epoch 120. The four rates tried are within $5,216 of each other on fold 0, less than Task 1A's fold-to-fold s.d. of $11,044, so fold 0 alone does not separate them. Fix the learning rate and epoch budget together before 2A3, 2B and 2C.
