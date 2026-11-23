"""Task 2B1: the cross-attention network with its MLP part initialised from the 1990 weights,
against the 2A2 baseline (PyTorch's default initialisation). Everything else is 2A2's setting.

2A2's own run() is reused, so training, recording and saving are identical; the only change
is Config(init="1990"). attention_model_fn copies the 1990 weights inside fork_rng(), so the
attention weights W_Q, W_K, W_V and the random state are the same for both runs (checked in
t2a1_build_check.py): the comparison changes exactly one thing.

Both initialisations are trained here, in parallel processes, so both curves come from the
same machine. The default run should reproduce results/t2a2_overfit.json at the same
learning rate; the script prints the difference as a check.

    python -m experiments.t2b_transfer_init                     (~10 min, 2 processes)
    python -m experiments.t2b_transfer_init --from-saved        (summary + figure only)
    python -m experiments.t2b_transfer_init --epochs 30         (smoke test, ~1 min)

Writes results/t2b1_transfer_init.json (+ .png, .csv via report.render) and
results/t2b1_transfer_init_overlay.png. 2B2 (+ dropout, LoRA) waits for 2A3 and apply_lora().
"""
import argparse
import json
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.t2a2_overfit import RECORD_EVERY, ZOOM_EPOCHS, run, summarise
from housing.attention import D_K
from housing.report import RESULTS, save_result
from housing.train import Config

NAME: str = "t2b1_transfer_init"
BASELINE: str = "t2a2_overfit"
LABELS: dict[str, str] = {"default": "default init (2A2 baseline)", "1990": "1990 weights in the MLP part"}


def saved_baseline(cfg: Config) -> dict | None:
    """The 2A2 run with this setting, if 2A2's results are present."""
    path = RESULTS / f"{BASELINE}.json"
    if not path.exists():
        return None
    for r in json.loads(path.read_text())["runs"]:
        if r["config"] == Config(**{**cfg.to_dict(), "init": "default"}).to_dict():
            return r
    return None


def overlay(runs: list[dict], path) -> None:
    """Both initialisations: validation solid, training dashed, log MAE; left panel zooms in."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, limit in zip(axes, (ZOOM_EPOCHS, None)):
        for i, r in enumerate(runs):
            c, curves, label = f"C{i}", r["curves"], LABELS[r["config"]["init"]]
            ax.plot(curves["epoch"], curves["validation_mae"], "-", color=c, lw=1.4, label=f"{label} (validation)")
            ax.plot(curves["epoch"], curves["train_mae"], "--", color=c, lw=1.0, label=f"{label} (training)")
            b = r["best_validation"]
            ax.plot(b["epoch"], b["validation_mae"], "o", color=c, ms=4)
        ax.set(xlabel="epoch (one full-batch Adam step)", yscale="log",
               xlim=(0, limit) if limit else None,
               title=f"epochs 0-{limit} (dots: best validation)" if limit else "all epochs")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE ($)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    cfg = runs[0]["config"]
    fig.suptitle(f"2B1: cross-attention (d_k={D_K}), lr {cfg['lr']:g}, {cfg['loss'].upper()} loss, "
                 f"no dropout, validation = fold 0")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report(runs: list[dict]) -> None:
    summary = summarise(runs)
    for s, r in zip(summary, runs):
        s["init"] = r["config"]["init"]
    print(f"{'init':>8} | {'epoch 0 val':>11} | {'best val':>10} {'epoch':>6} | "
          f"{'final train':>11} {'mean last 10':>12} {'final val':>10}")
    for s, r in zip(summary, runs):
        print(f"{s['init']:>8} | {r['curves']['validation_mae'][0]:>11,.0f} | {s['best_validation_mae']:>10,.0f} "
              f"{s['best_epoch']:>6} | {s['final_train_mae']:>11,.0f} {s['mean_train_mae_last_10_records']:>12,.0f} "
              f"{s['final_validation_mae']:>10,.0f}   ({s['status']})")

    default = next(r for r in runs if r["config"]["init"] == "default")
    saved = saved_baseline(Config(**default["config"]))
    if saved and len(saved["curves"]["epoch"]) == len(default["curves"]["epoch"]):
        gap = max(abs(a - b) for a, b in zip(saved["curves"]["validation_mae"], default["curves"]["validation_mae"]))
        print(f"check: default run vs saved 2A2 run, largest validation-curve gap ${gap:,.0f} "
              f"(0 on the machine 2A2 was recorded on; small elsewhere)")

    path = save_result(NAME, {"task": "2B1", "validation_fold": 0,
                              "model": {"name": "CrossAttentionRegressor", "d_k": D_K,
                                        "record_every": RECORD_EVERY, "baseline": BASELINE},
                              "runs": runs, "summary": summary})
    overlay(runs, RESULTS / f"{NAME}_overlay.png")
    print(path)


def main() -> None:
    p = argparse.ArgumentParser(description="2B1: transferred initialisation of the MLP part")
    p.add_argument("--lr", type=float, default=1e-3, help="the group's Task 2 base rate (TASK_2A.md)")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--loss", default="l1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-saved", action="store_true", help="rebuild summary and figure, no training")
    a = p.parse_args()

    if a.from_saved:
        runs = json.loads((RESULTS / f"{NAME}.json").read_text())["runs"]
    else:
        configs = [Config(lr=a.lr, epochs=a.epochs, loss=a.loss, seed=a.seed, init=init)
                   for init in ("default", "1990")]
        with Pool(2) as pool:
            runs = pool.map(run, configs, chunksize=1)
    report(runs)


if __name__ == "__main__":
    main()
