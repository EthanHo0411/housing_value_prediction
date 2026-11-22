# Task 2: look-out notes for P4 (2A3 dropout, LoRA, 2C)

From P3. The cross-attention network exists on branch `task-2a`; your parts plug into it.
Read this before writing code: most of it is about keeping your results comparable with
2A2, 2A4, 2B and the 2D ablation table.

## 1. What already exists

| File | What it gives you |
|---|---|
| `housing/attention.py` | `CrossAttentionRegressor`, `attention_model_fn(cfg, d_k=64)`, `position_encoding()`, `source_parts()` |
| `experiments/t2a1_build_check.py` | 23 checks: 17,089 parameters, token shapes, seeds and random state, gradients, batching (~10 s) |
| `experiments/t2a2_overfit.py` | 2A2 and the Task 2 base learning rate (`results/t2a2_overfit.json`, key `selected`) |

The forward pass, in the spec's notation, with the method that computes each line:

```
h2 = mlp.hidden(x_std)                          MLP's 2nd hidden layer (dropout already applied in training)
r  = receiving_token(h2, x_raw)                 h2 + phi(2020)                            (N, 64)
z  = source_tokens()                            [g2_1990 + phi_1990 ; ybar_1990]          (20640, 65)
a  = attention_weights(q_proj(r), k_proj(z))    softmax over the 1990 records / sqrt(d_k) (N, 20640)
c  = a @ v_proj(z)                                                                        (N, 64)
y  = mlp.out(h2 + c)
```

## 2. Where your parts plug in (without editing `attention.py`)

Put your code in a new file, e.g. `housing/attention_extras.py`, and subclass or wrap.
Two people editing `attention.py` at once will conflict, and a change there changes 2A2's numbers.
If something in it has to change, tell P3 first.

| Your part | How | Watch out |
|---|---|---|
| **2A3: dropout on MLP hidden units** | Already there: `Config(drop_rate=p)`. `h2` is dropped *before* it enters both `r` and `h2 + c`. | `drop_rate` is the probability of **dropping** (PyTorch), not keeping (Week 4 slides). |
| **2A3: dropout on attention weights** | Subclass, override `attention_weights()`: `F.dropout(super().attention_weights(q, k), p, self.training)` | Inverted dropout rescales by 1/(1-p): rows no longer sum to 1 while training. Fine, but say it. |
| **2A3: dropout on entries of W_Q, W_K, W_V** | Keep the existing layers and add a parametrization: `with torch.random.fork_rng(): torch.nn.utils.parametrize.register_parametrization(model.q_proj, "weight", nn.Dropout(p))` (same for `k_proj`, `v_proj`) | **Do not replace** `q_proj` with a new `Linear` subclass: a new layer redraws `W_Q`, so the run changes initialisation as well as dropout (checked). The parametrization keeps 2A2's weights and the 17,089 count, zeroes about p of the entries per training forward pass, and is a no-op in `eval()`. Registering it calls the dropout once, so wrap it in `fork_rng()` as shown. |
| **LoRA (P3's 2A4 waits on this)** | Provide `apply_lora(model, r=4) -> model` replacing `q_proj`, `k_proj`, `v_proj` with a LoRA layer: frozen base `W0` (the existing `Linear`), `B` (in x r) **zeros**, `A` (r x out) PyTorch default init; output `base(x) + (x @ B) @ A` | Spec: `W = W0 + B A`, `B` = 0 at the start, `W0` frozen at its (random) default init. nn.Linear stores `W^T` (out x in); mind the orientation. |
| **2C, C1.i: price encoding** | Subclass; build it with `functools.partial(attention_model_fn, model_cls=YourSubclass, source_dim=128)`; override `source_tokens()` to return `[g2 + phi ; psi(ybar)]`, psi = 32 frequencies log-spaced 0.5 to 32, `sin(2 pi f ybar)`, `cos(2 pi f ybar)` | `ybar` is the buffer `self.ybar_1990` (N, 1), already in [0, 1]. |
| **2C, C1.ii: layer norm** | Subclass (build with `functools.partial(attention_model_fn, model_cls=YourSubclass)`); override `receiving_token()` -> `LN(h2) + LN(phi)` and `source_tokens()` -> `[LN(g2) + LN(phi) ; ybar]`, four separate `nn.LayerNorm(64)` | `g2_1990` and `phi_1990` are separate buffers so you can normalise them separately. |

**How to build a variant so seeds stay fair:** do it inside the model function, after the
base network, e.g. `model_fn = lambda cfg: apply_lora(attention_model_fn(cfg), r=4)`, or
for a subclass `functools.partial(attention_model_fn, model_cls=YourSubclass, ...)`, which
also keeps the `init="1990"` path (2B) working for your variant. `train_full_batch` seeds
before calling it, so the MLP and `W0` start exactly as in 2A2, and only your new parameters
are extra. New parameters (LoRA's `A`) consume random numbers after that, so their dropout
masks differ from 2A2's; that is expected, but say it. Anything that builds a network only to
read data from it (as `source_parts()` does with the 1990 network), or that calls a dropout
while setting up, goes inside `torch.random.fork_rng()`, or it shifts every later random number.

## 3. Parameter counts to check (trainable)

| Model | Trainable | Breakdown |
|---|---|---|
| 2A2 baseline (`d_k` 64) | 17,089 | MLP 4,673 + W_Q 4,096 + W_K 4,160 + W_V 4,160 |
| + LoRA r = 4 | 6,217 | MLP 4,673 + Q 4(64+64) = 512 + K 4(65+64) = 516 + V 516; the 12,416 in `W0` frozen |
| C1.i price encoding | 25,153 | MLP 4,673 + W_Q 4,096 + W_K 8,192 + W_V 8,192 |
| C1.ii layer norm | 17,601 | 17,089 + 4 x 128 (gamma, beta per `LayerNorm(64)`) |

Use `housing.mlp.count_parameters(model)`. At the first LoRA step `A` gets a zero gradient
(it flows through `B` = 0): expected, not a bug.

## 4. Conventions (2D needs every row trained the same way)

- `Config` as in `configs/task1a_baseline.json`: L1 loss, seed 42; for CV `fold_seed_stride=1`
  and report fold spread with `summary(ddof=1)`. Call `configure_runtime()` (1 thread, deterministic).
- 2A and 2B: one split, **fold 0 as validation** (as 1A1, 2A2). 2C: five-fold CV (spec).
- Learning rate and epochs: proposed **lr 0.001**, Task 1A and 1B's rate, so 2D's rows can
  share it down to the Task 1A MLP row. On fold 0 its validation MAE is lowest at epoch 120
  (`results/t2a2_overfit.json` -> `summary`). The four rates tried are within $5,216 of each
  other there, less than Task 1A's fold-to-fold s.d. ($11,044). The group fixes the learning
  rate and epoch budget together before 2A3; see `TASK_2A.md`.
- `record_every=10` for Task 2 curves (a full MAE pass over 400 rows every epoch costs time).
- Dropout is switched off for measuring by the shared code (`eval()`), so no extra work there.

## 5. Pitfalls

- **Test set:** only in 2D's final run, through `load_test()`, with `eval_batch` (e.g. 2000):
  all 21,282 rows at once need a 1.76 GB attention matrix.
- **Run time:** one 2000-epoch run is minutes, a five-fold CV is five times that. Run
  settings in parallel processes (see `t2a2_overfit.py`, `multiprocessing.Pool`), each one thread.
- **The 1990 tokens are fixed:** dropout or layer norm cannot change `g2` inside the 1990
  network (it is computed once, in eval mode); normalise or drop on the buffers instead.
- **Never modify the 1990 buffers in place** (`self.g2_1990 += ...`, `.mul_()`, ...): every
  model in a process shares the same cached tensors, so an in-place edit corrupts all of them.
  Compute new tensors instead.
- **Results in git:** `results/*` is ignored except an allow-list in `.gitignore`. To share
  a final result without editing `.gitignore` (and risking a merge conflict), track it with
  `git add -f results/<file>`, as `task-2a` does for 2A2.
- **Merge conflicts:** keep your branch to new files where you can (as `task-2a` does). Edits
  to `README.md`, `.gitignore` or `experiments/README.md` collide with other branches' edits.
- **Before pushing:** `python checks.py`, `python -m experiments.t2a1_build_check`, your
  parameter counts, and a 3-epoch smoke run of each variant.

## 6. Note for P2 (2B)

Use `attention_model_fn` with `Config(init="1990")`: it copies the 1990 weights into the MLP
inside `fork_rng()`, so attention weights and dropout masks match the `init="default"` run
exactly (checked in `t2a1_build_check.py`). The Task 1 `default_model_fn` does not do this:
in 1B2, fine-tune and scratch draw different dropout masks.
