"""Task 2A2: train the cross-attention network from scratch (d_k = 64, PyTorch's default
initialisation) on folds 1-4, validate on fold 0, and look for overfitting.

Every run follows Task 1A's conventions (configs/task1a_baseline.json): L1 loss on the
standardised price, seed 42, one thread, deterministic algorithms; no dropout here (2A3 adds
it). Curves are recorded every 10 epochs. Each learning rate runs in its own process.
The script records facts (each rate's best epoch); the Task 2 base setting is a group
decision, see TASK_2A.md.

    python -m experiments.t2a2_overfit                                   (~10 min, 4 processes)
    python -m experiments.t2a2_overfit --from-saved                      (summary + figures only)
    python -m experiments.t2a2_overfit --learning-rates 1e-3 --epochs 500

Writes results/t2a2_overfit.json (+ .png, .csv via report.render) and t2a2_overfit_overlay.png.
"""
import argparse
import json
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from housing.attention import D_K, attention_model_fn
from housing.data import fold_split
from housing.report import RESULTS, fit_result, save_result
from housing.train import Config, configure_runtime, train_full_batch

NAME: str = "t2a2_overfit"
RECORD_EVERY: int = 10
ZOOM_EPOCHS: int = 400  # the validation minima all fall before this


def run(cfg: Config) -> dict:
    configure_runtime()
    train, val = fold_split(0)
    fit = train_full_batch(cfg, train, val, attention_model_fn, record_every=RECORD_EVERY)
    h = fit.history
    best = min(range(len(h.val_mae)), key=h.val_mae.__getitem__)
    result = fit_result(cfg, fit, len(train), len(val))
    result["best_validation"] = {"epoch": h.epoch[best], "validation_mae": h.val_mae[best],
                                 "train_mae": h.train_mae[best]}
    return result


def summarise(runs: list[dict]) -> list[dict]:
    """Per learning rate: best validation (epoch), final values, and the mean training MAE of the
    last 10 records (single late training readings are noisy at high learning rates)."""
    rows = []
    for r in runs:
        c, b = r["curves"], r["best_validation"]
        rows.append({"lr": r["config"]["lr"], "best_epoch": b["epoch"], "best_validation_mae": b["validation_mae"],
                     "final_train_mae": c["train_mae"][-1], "final_validation_mae": c["validation_mae"][-1],
                     "mean_train_mae_last_10_records": sum(c["train_mae"][-10:]) / len(c["train_mae"][-10:]),
                     "validation_rise_after_best": c["validation_mae"][-1] - b["validation_mae"],
                     "status": r["status"]})
    return rows


def overlay(runs: list[dict], path) -> None:
    """All learning rates: validation solid, training dashed, log MAE; left panel zooms on the
    first ZOOM_EPOCHS epochs where the validation minima are."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, limit in zip(axes, (ZOOM_EPOCHS, None)):
        for i, run in enumerate(runs):
            c, curves = f"C{i}", run["curves"]
            label = f"lr {run['config']['lr']:g}"
            ax.plot(curves["epoch"], curves["validation_mae"], "-", color=c, lw=1.4, label=f"{label} (validation)")
            ax.plot(curves["epoch"], curves["train_mae"], "--", color=c, lw=1.0, label=f"{label} (training)")
            b = run["best_validation"]
            ax.plot(b["epoch"], b["validation_mae"], "o", color=c, ms=4)
        ax.set(xlabel="epoch (one full-batch Adam step)", yscale="log",
               xlim=(0, limit) if limit else None,
               title=f"epochs 0-{limit} (dots: best validation)" if limit else "all 2000 epochs")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE ($)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"2A2: cross-attention (d_k={D_K}), default init, L1 loss, no dropout, validation = fold 0")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report(runs: list[dict]) -> None:
    summary = summarise(runs)
    print(f"{'lr':>8} | {'best val':>10} {'epoch':>6} | {'final train':>11} {'mean last 10':>12} {'final val':>10} | rise")
    for s in summary:
        print(f"{s['lr']:>8g} | {s['best_validation_mae']:>10,.0f} {s['best_epoch']:>6} | {s['final_train_mae']:>11,.0f} "
              f"{s['mean_train_mae_last_10_records']:>12,.0f} {s['final_validation_mae']:>10,.0f} | "
              f"{s['validation_rise_after_best']:>+8,.0f}   ({s['status']})", flush=True)
    path = save_result(NAME, {"task": "2A2", "validation_fold": 0,
                              "model": {"name": "CrossAttentionRegressor", "d_k": D_K, "init": "default",
                                        "record_every": RECORD_EVERY},
                              "runs": runs, "summary": summary})
    overlay(runs, RESULTS / f"{NAME}_overlay.png")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="2A2: overfitting of the cross-attention network")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--loss", default="l1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--from-saved", action="store_true", help="rebuild summary and figures, no training")
    args = parser.parse_args()

    if args.from_saved:
        runs = json.loads((RESULTS / f"{NAME}.json").read_text())["runs"]
    else:
        configs = [Config(lr=lr, epochs=args.epochs, loss=args.loss, seed=args.seed) for lr in args.learning_rates]
        with Pool(min(args.processes, len(configs))) as pool:
            runs = pool.map(run, configs, chunksize=1)
    report(runs)


if __name__ == "__main__":
    main()
