# Task 1B: domain adaptation and transfer

Two thin scripts over the shared library; no changes to `housing/`.

| Script | Sub-task | Shared API used |
|---|---|---|
| `experiments/t1b1_moment_matching.py` | 1B1 | `cross_validate(Config(lr=0, epochs=0, init="1990"))`, and `std_fn=Standardiser.from_json` for the unadapted baseline |
| `experiments/t1b2_finetune.py` | 1B2 | `cross_validate(Config(..., init="1990" / "default"), fold_seed_stride=1)` |

## Commands (from the repository root, after `python checks.py`)

| Report item | Command | Output in `results/` | Time |
|---|---|---|---|
| 1B1 table | `python -m experiments.t1b1_moment_matching` | `t1b1_moment_matching*.json` | ~3 s |
| 1B2 main figure + table | `python -m experiments.t1b2_finetune` | `t1b2_finetune.png`, `t1b2_finetune_{finetune,scratch}.json` | ~15 s |
| 1B2 longer budget | `python -m experiments.t1b2_finetune --long` | `t1b2_finetune_long.png`, `t1b2_finetune_long_*.json` | ~1 min |
| 1B2 settings grid | `python -m experiments.t1b2_finetune --sweep` | `t1b2_finetune_sweep.json` | ~4 min |

## Protocol

- The 2020 statistics are fitted on the four training folds of each round (shared default), with population s.d.
- 1B2 uses **Task 1A's final settings** (A5): L1 loss, dropout 0.1, lr 1e-3, 500 epochs, seed 42 with seed 42 + k in fold k (`fold_seed_stride=1`), one thread, deterministic algorithms (`housing.task1a.setup`). Fine-tune and scratch differ only in `init`.
- CV MAE is at the final epoch (fixed budget), mean ± sample s.d. (ddof=1) over the folds, as in the 1A tables.
- Epoch 0 of every fine-tuning fold is the 1B1 moment-matched model (also checked in `checks.py`).

## Results (5-fold CV validation MAE, $)

Recorded on Linux, CPU, torch 2.15.0, Python 3.13. The scratch run at 1A settings gave
157,515 ± 11,019 here vs 157,609 ± 11,044 recorded for 1A4 (dropout 0.1): same seeds and
code, so the ~$100 gap comes from numerical differences between machines/builds.
Rerun on the lab machine before quoting numbers.

### 1B1: moment matching

| Model | CV MAE |
|---|---|
| 1990 network, 1990 statistics (no adaptation) | 280,747 ± 13,262 |
| 1990 network, 2020 statistics (moment matching) | 207,049 ± 12,189 |
| Task 1A selected MLP (L1, dropout 0.1), from 1A4 | 157,609 ± 11,044 |

Moment matching takes about $74k off the error of the unadapted network, but is about $49k
worse than the 1A MLP trained on 400 examples from 2020. It only shifts and rescales each
variable separately, so it cannot capture changes in how the features relate to price.

### 1B2: fine-tuning vs from scratch (1A settings, 500 epochs)

| Initialisation | Epoch 0 | Final (500) | Final train MAE |
|---|---|---|---|
| 1990 weights (fine-tune) | 207,049 ± 12,189 | 163,278 ± 5,420 | 98,687 |
| PyTorch default (scratch) | 290,936 ± 15,780 | 157,515 ± 11,019 | 112,514 |

- Fine-tuning starts about $84k lower, but scratch's mean validation curve is below it from epoch 28 on.
- Fine-tuning ends $5,763 worse on average (worse on 4 of 5 folds), with *lower* training MAE: the 1990 starting point fits the training rows more easily, but generalises worse.
- **Grid** (`--sweep`, 500 epochs, lr ∈ {1e-4, 3e-4, 1e-3} × dropout ∈ {0, 0.1, 0.2, 0.5} × {MSE, L1}): fine-tuning ends worse in **24 of 24** settings. The gap is smallest at lr 1e-3 with dropout 0.1–0.2 (+$4–6k), and largest with dropout 0.5 (+$26k to +$124k): the 1990 weights were trained without dropout.
- **Longer budget** (`--long`, 2000 epochs): scratch overfits after about epoch 500 (final 167,059 ± 8,527) and fine-tuning ends lower (162,259 ± 9,434), but only on 2 of 5 folds, with per-fold differences from −$30k to +$15k: no consistent effect. Both are worse than their best epochs (157,507 at 501 for scratch, 158,510 at 834 for fine-tuning), which is why 1A chose a 500-epoch budget.

**Does the transferred initialisation help?** No. It gives a better starting point, but
training on the 2020 data overtakes it within about 30 epochs, and at the settings chosen
in Task 1A it ends worse.
