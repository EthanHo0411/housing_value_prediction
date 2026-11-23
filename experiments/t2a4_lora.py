"""Task 2A4: LoRA (r = 4) on W_Q, W_K, W_V, without and with dropout.

Same protocol as 2A2 at lr 0.001 (TASK_2A.md): d_k = 64, default initialisation, L1 loss,
seed 42, folds 1-4 train, fold 0 validation, 2000 full-batch Adam steps, MAE every 10 epochs,
one thread per run, runs in parallel processes (each run re-seeds). LoRA and the dropout options come from P4's
housing/attention_extras.py. The comparison without LoRA is the recorded 2A2 run at lr 0.001
(results/t2a2_overfit.json), not retrained.

    (1) LoRA, no dropout                      does LoRA reduce overfitting compared with 2A2?
    (2) LoRA + dropout 0.1 on the hidden units, on the attention weights, on the entries of
        W_Q, W_K, W_V                         can LoRA be used together with dropout?

    python -m experiments.t2a4_lora                (~20 min, 4 processes)
    python -m experiments.t2a4_lora --from-saved   (table + figure only)

Writes results/t2a4_lora.json, results/t2a4_lora.csv and results/t2a4_lora_overlay.png.
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
from housing.attention_extras import LORA_RANK, make_model_fn
from housing.data import fold_split
from housing.mlp import count_parameters
from housing.report import RESULTS, fit_result
from housing.train import Config, configure_runtime, train_full_batch

NAME: str = "t2a4_lora"
LR: float = 1e-3
EPOCHS: int = 2000
RECORD_EVERY: int = 10
DROP: float = 0.1  # the Task 1A baseline dropout rate (configs/task1a_baseline.json)
ZOOM_EPOCHS: int = 400  # 2A2's validation minima all fall before this

# (label, Config.drop_rate, make_model_fn options)
SETTINGS: list[tuple[str, float, dict]] = [
    ("LoRA", 0.0, {"lora_rank": LORA_RANK}),
    (f"LoRA + hidden dropout {DROP:g}", DROP, {"lora_rank": LORA_RANK}),
    (f"LoRA + attention dropout {DROP:g}", 0.0, {"lora_rank": LORA_RANK, "attn_drop": DROP}),
    (f"LoRA + weight dropout {DROP:g}", 0.0, {"lora_rank": LORA_RANK, "weight_drop": DROP}),
]


def best(curves: dict) -> dict:
    i = min(range(len(curves["validation_mae"])), key=curves["validation_mae"].__getitem__)
    return {"epoch": curves["epoch"][i], "validation_mae": curves["validation_mae"][i],
            "train_mae": curves["train_mae"][i]}


def run(setting: tuple[str, float, dict]) -> dict:
    label, drop_rate, options = setting
    configure_runtime()
    cfg = Config(lr=LR, epochs=EPOCHS, drop_rate=drop_rate, loss="l1", seed=42)
    train, val = fold_split(0)
    fit = train_full_batch(cfg, train, val, make_model_fn(**options), record_every=RECORD_EVERY)
    result = fit_result(cfg, fit, len(train), len(val))
    result.update(label=label, options=options, trainable_parameters=count_parameters(fit.model),
                  best_validation=best(result["curves"]))
    return result


def baseline_2a2() -> dict:
    """The recorded 2A2 run at the same learning rate: the network without LoRA or dropout."""
    runs = json.loads((RESULTS / "t2a2_overfit.json").read_text())["runs"]
    r = next(r for r in runs if r["config"]["lr"] == LR)
    params = count_parameters(attention_model_fn(Config.from_dict(r["config"])))
    return {**r, "label": "no LoRA (2A2)", "options": {}, "trainable_parameters": params,
            "best_validation": best(r["curves"])}


def summarise(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        c, b = r["curves"], r["best_validation"]
        last_train = c["train_mae"][-10:]
        rows.append({"label": r["label"], "trainable_parameters": r["trainable_parameters"],
                     "best_epoch": b["epoch"], "best_validation_mae": b["validation_mae"],
                     "train_mae_at_best": b["train_mae"], "gap_at_best": b["validation_mae"] - b["train_mae"],
                     "final_validation_mae": c["validation_mae"][-1],
                     "validation_rise_after_best": c["validation_mae"][-1] - b["validation_mae"],
                     "final_train_mae": c["train_mae"][-1],
                     "mean_train_mae_last_10_records": sum(last_train) / len(last_train),
                     "status": r["status"]})
    return rows


def overlay(runs: list[dict], path: Path) -> None:
    """Validation (solid) and training (dashed) MAE of every run, log scale; the left panel zooms
    on the first ZOOM_EPOCHS epochs, as in 2A2."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, limit in zip(axes, (ZOOM_EPOCHS, None)):
        for i, r in enumerate(runs):
            c, b, colour = r["curves"], r["best_validation"], f"C{i}"
            ax.plot(c["epoch"], c["validation_mae"], "-", color=colour, lw=1.4, label=f"{r['label']} (validation)")
            ax.plot(c["epoch"], c["train_mae"], "--", color=colour, lw=1.0, label=f"{r['label']} (training)")
            ax.plot(b["epoch"], b["validation_mae"], "o", color=colour, ms=4)
        ax.set(xlabel="epoch (one full-batch Adam step)", yscale="log", xlim=(0, limit) if limit else None,
               title=f"epochs 0-{limit} (dots: best validation)" if limit else f"all {EPOCHS} epochs")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE ($)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"2A4: LoRA r={LORA_RANK} on W_Q, W_K, W_V (d_k={D_K}), lr {LR:g}, L1, validation = fold 0")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report(runs: list[dict]) -> None:
    runs = [baseline_2a2()] + runs
    summary = summarise(runs)
    print(f"{'run':<30} {'params':>7} | {'best val':>9} {'epoch':>5} | {'final val':>9} {'rise':>8} | "
          f"{'final train':>11} {'mean last 10':>12}")
    for s in summary:
        print(f"{s['label']:<30} {s['trainable_parameters']:>7,} | {s['best_validation_mae']:>9,.0f} "
              f"{s['best_epoch']:>5} | {s['final_validation_mae']:>9,.0f} {s['validation_rise_after_best']:>+8,.0f} | "
              f"{s['final_train_mae']:>11,.0f} {s['mean_train_mae_last_10_records']:>12,.0f}  ({s['status']})",
              flush=True)
    path = RESULTS / f"{NAME}.json"
    path.write_text(json.dumps({"task": "2A4", "validation_fold": 0,
                                "baseline": {"file": "t2a2_overfit.json", "lr": LR, "summary_row": 0},
                                "model": {"name": "CrossAttentionRegressor + attention_extras (LoRA)", "d_k": D_K,
                                          "lora_rank": LORA_RANK, "init": "default", "record_every": RECORD_EVERY},
                                "runs": runs[1:], "summary": summary}, indent=2, allow_nan=False) + "\n")
    with open(RESULTS / f"{NAME}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    overlay(runs, RESULTS / f"{NAME}_overlay.png")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="2A4: LoRA without and with dropout")
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
