"""Task 2A3: dropout in three places of the cross-attention network, at two rates.

Same protocol as 2A2 at lr 0.001 (TASK_2A.md): d_k = 64, default initialisation, L1 loss,
seed 42, folds 1-4 train, fold 0 validation, 2000 full-batch Adam steps, MAE every 10 epochs,
one thread per run, runs in parallel processes (each run re-seeds). Only the dropout changes;
the comparison without dropout is the recorded 2A2 run at lr 0.001 (results/t2a2_overfit.json).

    hidden units      Config(drop_rate=p): the MLP's dropout after both hidden layers
    attention weights make_model_fn(attn_drop=p): on a_ij after the softmax
    weight entries    make_model_fn(weight_drop=p): on the entries of W_Q, W_K, W_V (DropConnect)

The attention and weight options come from P4's housing/attention_extras.py. Every placement
starts from the 2A2 initial weights (checked in experiments/t2_extras_check.py).

    python -m experiments.t2a3_dropout                (~25 min, 6 processes)
    python -m experiments.t2a3_dropout --from-saved   (table + figure only)

Writes results/t2a3_dropout.json, results/t2a3_dropout.csv and results/t2a3_dropout_overlay.png.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from housing.attention import D_K, attention_model_fn
from housing.attention_extras import make_model_fn
from housing.data import fold_split
from housing.mlp import count_parameters
from housing.report import RESULTS, fit_result
from housing.train import Config, configure_runtime, train_full_batch

NAME: str = "t2a3_dropout"
LR: float = 1e-3
EPOCHS: int = 2000
RECORD_EVERY: int = 10
RATES: tuple[float, ...] = (0.1, 0.3)
ZOOM_EPOCHS: int = 400  # 2A2's validation minima all fall before this

# (label, placement, Config.drop_rate, make_model_fn options)
SETTINGS: list[tuple[str, str, float, dict]] = (
    [(f"hidden {p:g}", "hidden", p, {}) for p in RATES]
    + [(f"attention {p:g}", "attention", 0.0, {"attn_drop": p}) for p in RATES]
    + [(f"weights {p:g}", "weights", 0.0, {"weight_drop": p}) for p in RATES]
)
COLOURS: dict[str, str] = {"none": "C0", "hidden": "C1", "attention": "C2", "weights": "C3"}


def best(curves: dict) -> dict:
    i = min(range(len(curves["validation_mae"])), key=curves["validation_mae"].__getitem__)
    return {"epoch": curves["epoch"][i], "validation_mae": curves["validation_mae"][i],
            "train_mae": curves["train_mae"][i]}


def run(setting: tuple[str, str, float, dict]) -> dict:
    label, placement, drop_rate, options = setting
    configure_runtime()
    cfg = Config(lr=LR, epochs=EPOCHS, drop_rate=drop_rate, loss="l1", seed=42)
    train, val = fold_split(0)
    fit = train_full_batch(cfg, train, val, make_model_fn(**options), record_every=RECORD_EVERY)
    result = fit_result(cfg, fit, len(train), len(val))
    result.update(label=label, placement=placement, options=options,
                  trainable_parameters=count_parameters(fit.model), best_validation=best(result["curves"]))
    return result


def baseline_2a2() -> dict:
    """The recorded 2A2 run at the same learning rate: the network without dropout."""
    runs = json.loads((RESULTS / "t2a2_overfit.json").read_text())["runs"]
    r = next(r for r in runs if r["config"]["lr"] == LR)
    params = count_parameters(attention_model_fn(Config.from_dict(r["config"])))
    return {**r, "label": "no dropout (2A2)", "placement": "none", "options": {},
            "trainable_parameters": params, "best_validation": best(r["curves"])}


def summarise(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        c, b = r["curves"], r["best_validation"]
        last_train = c["train_mae"][-10:]
        rows.append({"label": r["label"], "placement": r["placement"], "trainable_parameters": r["trainable_parameters"],
                     "best_epoch": b["epoch"], "best_validation_mae": b["validation_mae"],
                     "train_mae_at_best": b["train_mae"], "gap_at_best": b["validation_mae"] - b["train_mae"],
                     "final_validation_mae": c["validation_mae"][-1],
                     "validation_rise_after_best": c["validation_mae"][-1] - b["validation_mae"],
                     "final_train_mae": c["train_mae"][-1],
                     "mean_train_mae_last_10_records": sum(last_train) / len(last_train),
                     "status": r["status"]})
    return rows


def overlay(runs: list[dict], path: Path) -> None:
    """Validation MAE (left, first ZOOM_EPOCHS epochs; middle, all epochs) and training MAE (right).
    One colour per placement; rate 0.1 solid, 0.3 dashed."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for r in runs:
        c, b = r["curves"], r["best_validation"]
        rate = r["config"]["drop_rate"] or r["options"].get("attn_drop") or r["options"].get("weight_drop") or 0.0
        style = {"color": COLOURS[r["placement"]], "lw": 1.3, "ls": "--" if rate > 0.2 else "-"}
        for ax in axes[:2]:
            ax.plot(c["epoch"], c["validation_mae"], label=r["label"], **style)
            ax.plot(b["epoch"], b["validation_mae"], "o", color=style["color"], ms=3.5)
        axes[2].plot(c["epoch"], c["train_mae"], **style)
    titles = (f"validation, epochs 0-{ZOOM_EPOCHS} (dots: best)", f"validation, all {EPOCHS} epochs",
              "training (folds 1-4)")
    for ax, title, limit in zip(axes, titles, (ZOOM_EPOCHS, None, None)):
        ax.set(xlabel="epoch (one full-batch Adam step)", yscale="log", title=title,
               xlim=(0, limit) if limit else None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE ($)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"2A3: dropout placements, cross-attention (d_k={D_K}), lr {LR:g}, L1, validation = fold 0")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report(runs: list[dict]) -> None:
    runs = [baseline_2a2()] + runs
    summary = summarise(runs)
    print(f"{'run':<18} | {'best val':>9} {'epoch':>5} {'train@best':>10} | {'final val':>9} {'rise':>8} | "
          f"{'mean train last 10':>18}")
    for s in summary:
        print(f"{s['label']:<18} | {s['best_validation_mae']:>9,.0f} {s['best_epoch']:>5} {s['train_mae_at_best']:>10,.0f} | "
              f"{s['final_validation_mae']:>9,.0f} {s['validation_rise_after_best']:>+8,.0f} | "
              f"{s['mean_train_mae_last_10_records']:>18,.0f}  ({s['status']})", flush=True)
    path = RESULTS / f"{NAME}.json"
    path.write_text(json.dumps({"task": "2A3", "validation_fold": 0,
                                "baseline": {"file": "t2a2_overfit.json", "lr": LR, "summary_row": 0},
                                "model": {"name": "CrossAttentionRegressor + attention_extras", "d_k": D_K,
                                          "init": "default", "record_every": RECORD_EVERY},
                                "runs": runs[1:], "summary": summary}, indent=2, allow_nan=False) + "\n")
    with open(RESULTS / f"{NAME}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    overlay(runs, RESULTS / f"{NAME}_overlay.png")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="2A3: dropout placements")
    parser.add_argument("--from-saved", action="store_true", help="rebuild table and figure, no training")
    parser.add_argument("--processes", type=int, default=len(SETTINGS))
    args = parser.parse_args()
    if args.from_saved:
        runs = json.loads((RESULTS / f"{NAME}.json").read_text())["runs"]
    else:
        with Pool(min(args.processes, len(SETTINGS))) as pool:
            runs = pool.map(run, SETTINGS, chunksize=1)
    report(runs)


if __name__ == "__main__":
    main()
