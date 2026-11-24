# Task 2A: cross-attention (P3's part)

| Script | Sub-task | Status |
|---|---|---|
| `housing/attention.py` | 2A1 network | done |
| `experiments/t2a1_build_check.py` | 2A1 checks (~10 s; no test data, writes nothing) | done, all pass |
| `experiments/t2a2_overfit.py` | 2A2 overfitting, learning rates 0.0001-0.003 | done |
| `experiments/t2a3_dropout.py` | 2A3 dropout placements (P4's task, run with P4's `housing/attention_extras.py`) | done |
| `experiments/t2a4_lora.py` | 2A4 LoRA, without and with dropout (uses P4's `housing/attention_extras.py`) | done |

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
| 2A3 curves and table | `python -m experiments.t2a3_dropout` | `t2a3_dropout.json`, `.csv`, `t2a3_dropout_overlay.png` | ~17 min (6 processes) |
| 2A3 table and figure from saved runs | `python -m experiments.t2a3_dropout --from-saved` | same files, no training | seconds |
| 2A3 diagnostics: attention spread, h2 / c output paths | `python -m experiments.t2a3_diagnostics` | `t2a3_diagnostics.json` | ~15 min (2 processes) |
| 2A4 curves and table | `python -m experiments.t2a4_lora` | `t2a4_lora.json`, `.csv`, `t2a4_lora_overlay.png` | ~20 min (4 processes) |
| 2A4 table and figure from saved runs | `python -m experiments.t2a4_lora --from-saved` | same files, no training | seconds |

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

## 2A3: dropout placements (fold 0, same machine and versions as 2A2)

Same protocol as 2A2 at lr 0.001; only the dropout changes, in one place per run, at 0.1 and
0.3. Hidden units: `Config(drop_rate)`, after both MLP hidden layers. Attention weights and
weight entries: P4's `make_model_fn(attn_drop=...)`, `make_model_fn(weight_drop=...)`. Every run
starts from the 2A2 initial weights. The row without dropout is the recorded 2A2 run.

| Run | Best validation MAE (epoch) | Training MAE at best | Final validation MAE | Rise after best | Mean train MAE, last 10 records |
|---|---|---|---|---|---|
| No dropout (2A2) | 147,323 (120) | 116,700 | 200,922 | +53,599 | 7,933 |
| Hidden 0.1 | 147,683 (90) | 134,077 | 225,544 | +77,861 | 9,360 |
| Hidden 0.3 | 149,369 (100) | 138,093 | 255,724 | +106,354 | 7,249 |
| Attention 0.1 | 146,920 (120) | 116,258 | 201,610 | +54,689 | 7,458 |
| Attention 0.3 | 147,271 (120) | 117,410 | 205,482 | +58,211 | 7,351 |
| Weights 0.1 | 145,542 (120) | 133,247 | 183,511 | +37,970 | 27,837 |
| Weights 0.3 | 141,682 (170) | 133,303 | 168,143 | +26,460 | 43,771 |

- Dropout on the weight entries (DropConnect) is the only placement that reduces overfitting:
  the training MAE stays higher ($28k and $44k vs $8k) and the validation MAE rises less after
  its best epoch ($38k and $26k vs $54k), more so at 0.3.
- Dropout on the attention weights changes almost nothing: the curves overlap 2A2's.
- Dropout on the hidden units does not regularise here: the training MAE still reaches about
  $7k-$9k and the validation MAE rises more ($78k and $106k).
- One seed, one fold: the best validation values differ by less than Task 1A's fold-to-fold s.d.
  ($11,044), so the comparison rests on the curve shapes (rise after best, training MAE).

### 2A3 diagnostics: why (`experiments/t2a3_diagnostics.py`)

Two networks trained as above (lr 0.001, 2000 epochs, fold 0): no dropout, and hidden-unit
dropout 0.5 (the strongest rate, where the effect is clearest). Measured in eval mode on the
training (T) and validation (V) rows; `results/t2a3_diagnostics.json`.

| Run | Effective 1990 records per row, 1 / sum a_ij^2 (T / V) | Largest a_ij (T / V) | Change in c under attention dropout 0.5 (T / V) | s.d. of w3.h2 (T / V) | s.d. of w3.c (T / V) |
|---|---|---|---|---|---|
| No dropout | 1,008 / 836 | 0.011 / 0.011 | 18% / 11% | 0.43 / 0.49 | 0.72 / 0.89 |
| Hidden 0.5 | 1,249 / 780 | 0.009 / 0.013 | 20% / 22% | 0.06 / 0.06 | 0.97 / 1.35 |

- **Attention dropout does almost nothing** because each row's attention is spread over about
  800-1,250 records: dropping half of the a_ij (and doubling the rest) changes the context c by
  only 11-22%, since c averages over many records.
- **Hidden-unit dropout makes overfitting worse** because the network moves its prediction onto
  the attention path: the output's spread from h2 collapses (0.49 -> 0.06 on the validation rows)
  while the spread from c grows (0.89 -> 1.35). Dropout disrupts h2 but not c, so the undropped
  attention path carries (and memorises) the fit. Consistent with 2A4, where hidden dropout did
  help once LoRA restricted that path.
- Recorded on a different machine from the 2A3 table above (same versions); floating-point
  differences between machines grow over 2000 epochs (this machine's no-dropout run ends at
  $208,638 validation MAE vs 2A2's $200,922), so compare the diagnostics within their own rows.
  The values were recorded with an earlier version of the same code; the script itself has not
  been rerun since it was moved to its own file (see `provenance` in the JSON).

## 2A4: LoRA (fold 0, same machine and versions as 2A2)

Same protocol as 2A2 at lr 0.001. LoRA r = 4 on `W_Q`, `W_K`, `W_V` (`B` = 0 at the start, `W0`
frozen at its default initialisation): 6,217 trainable parameters instead of 17,089. Dropout rate
0.1 (Task 1A baseline) in one place per run. The row without LoRA is the recorded 2A2 run.

| Run | Best validation MAE (epoch) | Training MAE at best | Final validation MAE | Rise after best | Mean train MAE, last 10 records |
|---|---|---|---|---|---|
| No LoRA (2A2) | 147,323 (120) | 116,700 | 200,922 | +53,599 | 7,933 |
| LoRA | 141,358 (110) | 133,692 | 188,550 | +47,192 | 33,367 |
| LoRA + hidden dropout 0.1 | 139,902 (130) | 134,085 | 176,467 | +36,564 | 37,071 |
| LoRA + attention dropout 0.1 | 140,778 (110) | 133,789 | 162,797 | +22,019 | 30,003 |
| LoRA + weight dropout 0.1 | 136,047 (1300) | 66,465 | 144,705 | +8,658 | 58,721 |

- (1) LoRA reduces overfitting but does not remove it: the training MAE stays about 4x higher
  ($33k vs $8k), and the validation MAE rises less after its best epoch ($47k vs $54k), ending at
  $189k vs $201k.
- (2) LoRA and dropout work together: every placement reduces the rise further (hidden $37k,
  attention $22k, weight entries $9k). Dropout on the weight entries is the strongest: the
  validation curve stays flat to epoch 2000 (best at epoch 1300, final $145k).
- With LoRA, weight dropout drops entries of the merged `W0 + B A`, including the frozen `W0`.
- One seed, one fold: the best validation values differ by less than Task 1A's fold-to-fold
  s.d. ($11,044), so the conclusions rest on the curve shapes (rise after best, training MAE),
  not on the best values. The matching "dropout without LoRA" runs belong to 2A3 (P4).

## Base setting for the rest of Task 2 (group decision)

Proposed: **learning rate 0.001**, the rate of Task 1A and 1B, so every 2D ablation row can share it down to the Task 1A MLP row. On fold 0 its validation MAE is lowest at epoch 120. The four rates tried are within $5,216 of each other on fold 0, less than Task 1A's fold-to-fold s.d. of $11,044, so fold 0 alone does not separate them. Fix the learning rate and epoch budget together before 2A3, 2B and 2C.
