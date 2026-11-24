"""Task 2A3, diagnostics: why attention dropout does almost nothing and why hidden-unit
dropout makes overfitting worse, measured on trained networks. Complements
experiments/t2a3_dropout.py (the 2A3 comparison itself).

Two networks, trained exactly as in 2A2/2A3 (d_k = 64, default initialisation, L1 loss, lr 0.001,
seed 42, folds 1-4 train, fold 0 validation, 2000 full-batch Adam steps):

    no dropout        2A2's network (make_model_fn() with every option off)
    hidden 0.5        dropout 0.5 on the MLP hidden units (Config(drop_rate=0.5))

Then, in eval mode, on the training and the validation rows:

    attention spread   effective number of 1990 records per row, 1 / sum_j a_ij^2 (median over
                       rows), and the largest a_ij (median over rows)
    context change     ||c' - c|| / ||c|| (row norms, averaged over rows) where c' uses the a_ij
                       with dropout p = 0.5 applied (inverted, rescaled by 2), mean over 5 masks
                       from a fixed generator: how much attention dropout would perturb c
    output paths       y = w3.h2 + w3.c + b3; standard deviation over rows of the h2 part and of
                       the c part (standardised price units): which path the prediction uses

    python -m experiments.t2a3_diagnostics      (~15 min, 2 processes)

Writes results/t2a3_diagnostics.json.
"""
import json
from multiprocessing import Pool

import torch

from housing.attention_extras import make_model_fn
from housing.data import fold_split
from housing.report import RESULTS
from housing.train import Config, configure_runtime, train_full_batch

NAME: str = "t2a3_diagnostics"
LR: float = 1e-3
EPOCHS: int = 2000
SEED: int = 42
HIDDEN_RATE: float = 0.5   # the strongest hidden-unit dropout, where the effect is clearest
PROBE_RATE: float = 0.5    # attention-dropout rate used to probe the context change
N_MASKS: int = 5


def diagnose(drop_rate: float) -> dict:
    configure_runtime()
    cfg = Config(lr=LR, epochs=EPOCHS, drop_rate=drop_rate, loss="l1", seed=SEED)
    train, val = fold_split(0)
    fit = train_full_batch(cfg, train, val, make_model_fn(), record_every=EPOCHS)
    model, std = fit.model.eval(), fit.std
    gen = torch.Generator().manual_seed(0)
    out = {"run": "no dropout" if drop_rate == 0 else f"hidden {drop_rate:g}", "config": cfg.to_dict(),
           "final_train_mae": fit.history.train_mae[-1], "final_validation_mae": fit.history.val_mae[-1]}
    with torch.no_grad():
        z = model.source_tokens()
        k, v = model.k_proj(z), model.v_proj(z)
        for split, rows in (("train", train), ("validation", val)):
            x_std, x_raw = std.x(rows.x), torch.as_tensor(rows.x.copy())
            h2 = model.mlp.hidden(x_std)
            a = model.attention_weights(model.q_proj(model.receiving_token(h2, x_raw)), k)
            c = a @ v
            changes = [float(((a * (torch.rand(a.shape, generator=gen) >= PROBE_RATE) / (1 - PROBE_RATE)) @ v - c)
                             .norm(dim=1).mean() / c.norm(dim=1).mean()) for _ in range(N_MASKS)]
            w3 = model.mlp.out.weight[0]
            out[split] = {"effective_records_median": float((1 / (a * a).sum(1)).median()),
                          "max_attention_weight_median": float(a.max(1).values.median()),
                          f"context_change_under_attention_dropout_{PROBE_RATE:g}": sum(changes) / len(changes),
                          "output_sd_from_h2": float((h2 @ w3).std()), "output_sd_from_c": float((c @ w3).std())}
    return out


def main() -> None:
    with Pool(2) as pool:
        runs = pool.map(diagnose, [0.0, HIDDEN_RATE], chunksize=1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{NAME}.json"
    path.write_text(json.dumps({"task": "2A3 diagnostics", "validation_fold": 0, "epochs": EPOCHS,
                                "probe_attention_dropout": PROBE_RATE, "runs": runs}, indent=2, allow_nan=False) + "\n")
    key = f"context_change_under_attention_dropout_{PROBE_RATE:g}"
    for r in runs:
        print(f"{r['run']}: final train ${r['final_train_mae']:,.0f}, validation ${r['final_validation_mae']:,.0f}")
        for split in ("train", "validation"):
            d = r[split]
            print(f"  {split:<10} effective records {d['effective_records_median']:,.0f}, max a_ij "
                  f"{d['max_attention_weight_median']:.4f}, context change {d[key]:.3f}, "
                  f"output s.d. from h2 {d['output_sd_from_h2']:.3f} / from c {d['output_sd_from_c']:.3f}")
    print(path)


if __name__ == "__main__":
    main()
