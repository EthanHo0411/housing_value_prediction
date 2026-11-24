# Task 2D: ablation study and final evaluation

| Script | Sub-task | Status |
|---|---|---|
| `experiments/t2d_ablation.py` | 2D ablation table, five-fold CV | done |
| `experiments/t2d_ablation.py --final-test LABEL` | 2D final test (once) | done: cross-attention + LoRA + weight dropout |

## Commands (from the repository root)

| Report item | Command | Output in `results/` | Time |
|---|---|---|---|
| 2D ablation table and figure | `python -m experiments.t2d_ablation` | `t2d_ablation.json`, `.csv`, `.png` | ~40 min (2 processes) |
| 2D table and figure from saved runs | `python -m experiments.t2d_ablation --from-saved` | same files, no training | seconds |
| Smoke test | `python -m experiments.t2d_ablation --epochs 5` | `t2d_ablation_smoke.*` | ~1 min |
| 2D final test (ONCE) | `python -m experiments.t2d_ablation --final-test "<row label>"` | `t2d_final_test.json`, `t2d_model.pt` | ~5 min |

## Protocol

- Every row uses the Task 1A configuration (`configs/task1a_baseline.json`), as 2C does: lr 0.001,
  500 full-batch Adam steps, dropout 0.1 on the MLP hidden units, L1 loss, seed 42 + fold
  (`fold_seed_stride` 1), a fresh model and standardisation per fold, CV MAE at the final epoch,
  mean +/- sample s.d. (ddof 1). Same folds and seeds in every row.
- Rows: the Task 1A MLP, and cross-attention (d_k = 64) with every combination of LoRA (rank 4),
  weight dropout 0.1 (entries of W_Q, W_K, W_V; 0.1 is the group's rate) and transferred
  initialisation (MLP part from the 1990 weights): 9 rows, so any one-at-a-time path is covered.
- All rows trained on one machine (Windows, CPU, Python 3.14, torch 2.14.1, the lab versions).
  Cross-checks against rows recorded elsewhere: the MLP row gives $157,559 vs 1A4's $157,609 (two
  folds identical to the dollar), plain cross-attention $186,081 vs 2C's $185,078.

## Results (five-fold CV MAE, $)

| Row | Trainable | CV MAE | vs Task 1A MLP (folds better) |
|---|---|---|---|
| **Task 1A MLP** | 4,673 | **157,559 +/- 10,867** | -- |
| cross-attention + LoRA + weight dropout | 6,217 | 170,499 +/- 12,642 | +12,940 (0/5) |
| cross-attention + weight dropout | 17,089 | 173,974 +/- 12,758 | +16,414 (0/5) |
| cross-attention + LoRA | 6,217 | 181,897 +/- 22,510 | +24,338 (0/5) |
| cross-attention + LoRA + transfer | 6,217 | 183,512 +/- 24,089 | +25,952 (0/5) |
| cross-attention | 17,089 | 186,081 +/- 13,950 | +28,522 (0/5) |
| cross-attention + LoRA + weight dropout + transfer | 6,217 | 190,068 +/- 7,603 | +32,508 (0/5) |
| cross-attention + transfer | 17,089 | 205,030 +/- 15,819 | +47,471 (0/5) |
| cross-attention + weight dropout + transfer | 17,089 | 231,550 +/- 36,565 | +73,990 (0/5) |

### Ablation path (best cross-attention combination, one component removed at a time)

| Step | CV MAE | Change (paired over folds) |
|---|---|---|
| cross-attention + LoRA + weight dropout | 170,499 +/- 12,642 | |
| remove LoRA | 173,974 +/- 12,758 | +3,474 +/- 11,254 (worse on 3/5) |
| also remove weight dropout | 186,081 +/- 13,950 | +12,107 +/- 7,762 (worse on 5/5) |
| also remove cross-attention (Task 1A MLP) | 157,559 +/- 10,867 | -28,522 +/- 10,159 (better on 5/5) |

### Findings

- **No cross-attention combination beats the Task 1A MLP**: every one is worse on all five folds.
  Under these settings the best model is the MLP itself, i.e. attending to the 1990 records does
  not help.
- **Weight dropout contributes most** among the components: -$12.1k from plain cross-attention
  (better on 5/5 folds), -$11.4k with LoRA (4/5).
- **LoRA helps a little**: -$4.2k alone, -$3.5k with weight dropout, both within fold-to-fold noise
  (better on 3/5).
- **Transferred initialisation hurts** in every pairing: +$18.9k alone, +$57.6k with weight dropout,
  +$19.6k with LoRA and weight dropout (better on 0/5); with LoRA alone +$1.6k on average (better on
  3/5 folds), so no consistent gain.
  So it is left out of the best combination, consistent with 1B2 and 2B.
- **Cross-attention itself costs the most**: removing it improves the CV MAE by $28.5k.

### Limitation to state

The shared configuration includes dropout 0.1 on the MLP hidden units (Task 1A's choice). 2A3 found
that this kind of dropout makes the cross-attention network overfit more, so the setting may favour
the MLP row. The spec requires the same settings in every row; a different shared setting would
need every row, the MLP included, to be rerun.

## Final test (run once)

The best row by CV is the Task 1A MLP, whose test MAE already exists (1A5, the same
configuration). To compare Task 2 with Task 1 on the test set, the best cross-attention
combination by CV was also trained on all 500 examples (same settings as its CV row) and evaluated
once on the 21,282 test examples. The row was fixed by CV before the test set was read.

    python -m experiments.t2d_ablation --final-test "cross-attention + LoRA + weight dropout"

| Model | CV MAE | Test MAE |
|---|---|---|
| Task 1A MLP (1A5) | 157,609 +/- 11,044 | 161,795.67 |
| cross-attention + LoRA + weight dropout (2D) | 170,499 +/- 12,642 | **159,747.08** |

- On the test set the cross-attention model is $2,049 (1.3%) better than the Task 1A MLP, although
  CV ranked it $12.9k worse (on all five folds). The two measurements disagree in direction and
  the test difference is small, so the evidence is that cross-attention to the 1990 records is at
  best on a par with the 2020-only MLP, not a clear improvement.
- One difference between the two estimates: the final models train on all 500 examples, the CV
  models on 400; the extra 100 examples may help the larger network more.
- Written by the final test: `results/t2d_final_test.json`, `results/t2d_model.pt`.
