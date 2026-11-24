"""Task 1B2: fine-tune the 1990 network vs train from scratch, same settings, five-fold CV.

Fine-tune: Config(init="1990") starts from mlp_1990.pt; the 2020 statistics of each round
replace the 1990 ones, so epoch 0 of every fine-tuning fold is the 1B1 moment-matched model.
Scratch: Config(init="default"), everything else identical.

Main comparison: Task 1A's final settings (A5: L1, dropout 0.1, lr 1e-3, 500 epochs, seed 42,
seed 42 + k in fold k), so the scratch side reproduces the 1A4 run for dropout 0.1.

    python -m experiments.t1b2_finetune            main comparison + figure        (~1 min)
    python -m experiments.t1b2_finetune --long     same settings for 2000 epochs   (~3 min)
    python -m experiments.t1b2_finetune --sweep    lr x dropout x loss grid, 500 epochs (~4 min)
"""
import argparse
import itertools
from dataclasses import replace

import numpy as np

from housing.report import RESULTS_DIR, format_mae, load_run, save_run
from housing.train import configure_runtime
from housing.train import Config, CVResult, cross_validate

NAME = "t1b2_finetune"
MAIN = Config(lr=1e-3, epochs=500, drop_rate=0.1, loss="l1", seed=42)
STRIDE = 1  # Task 1A's seed protocol: fold k uses seed + k

GRID = {"lr": [1e-4, 3e-4, 1e-3], "drop_rate": [0.0, 0.1, 0.2, 0.5], "loss": ["mse", "l1"]}

LABELS = {"1990": "fine-tuned (1990 init)", "default": "from scratch"}


def run_pair(cfg: Config) -> dict[str, CVResult]:
    return {init: cross_validate(replace(cfg, init=init), fold_seed_stride=STRIDE)
            for init in ("1990", "default")}


def describe(pair: dict[str, CVResult]) -> None:
    for init, cv in pair.items():
        epochs, mean, _ = cv.mean_curve("val_mae")
        print(f"{LABELS[init]:>23}: epoch 0 ${format_mae(*cv.summary(0, ddof=1))}, "
              f"final ${format_mae(*cv.summary(ddof=1))}, "
              f"final train ${cv.mean_curve('train_mae')[1][-1]:,.0f}, "
              f"best mean val ${mean.min():,.0f} at epoch {epochs[mean.argmin()]}")
    d = pair["1990"].val_at() - pair["default"].val_at()
    print(f"per-fold final diff (fine-tune - scratch): {np.round(d).astype(int).tolist()}, "
          f"mean {d.mean():+,.0f}, fine-tune better on {(d < 0).sum()}/5 folds")
    ft_m, sc_m = pair["1990"].mean_curve("val_mae")[1], pair["default"].mean_curve("val_mae")[1]
    ahead = np.where(sc_m < ft_m)[0]
    if len(ahead):
        print(f"scratch's mean validation curve first drops below fine-tuning's at epoch {ahead[0]}")


def plot(pair: dict[str, CVResult], path, title: str) -> None:
    # Linear y-axis (the shared plot_curves defaults to log); same style otherwise.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (init, cv) in enumerate(pair.items()):
        c = f"C{i}"
        e, m, s = cv.mean_curve("val_mae")
        ax.plot(e, m / 1e3, "-", color=c, lw=1.5, label=f"{LABELS[init]} (validation)")
        ax.fill_between(e, (m - s) / 1e3, (m + s) / 1e3, color=c, alpha=0.15, lw=0)
        e, m, _ = cv.mean_curve("train_mae")
        ax.plot(e, m / 1e3, "--", color=c, lw=1.2, label=f"{LABELS[init]} (training)")
    ax.set(xlabel="epoch (one full-batch step)", ylabel="MAE (thousand $)", title=title, ylim=(0, 350))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main_comparison(cfg: Config, tag: str) -> None:
    pair = run_pair(cfg)
    for init, cv in pair.items():
        save_run(f"{NAME}{tag}_{'finetune' if init == '1990' else 'scratch'}", cv.config, cv.histories)
    print(f"Settings: {cfg.to_dict()}")
    describe(pair)
    title = (f"1B2: lr {cfg.lr:g}, dropout {cfg.drop_rate:g}, {cfg.loss.upper()} loss, "
             f"{cfg.epochs} epochs, 5-fold CV (band: ±1 s.d.)")
    plot(pair, RESULTS_DIR / f"{NAME}{tag}.png", title)


def sweep() -> None:
    rows = []
    print(f"Final-epoch ({MAIN.epochs}) CV MAE, $ mean +/- s.d. (ddof=1)")
    print(f"{'lr':>7} {'drop':>5} {'loss':>4} | {'scratch':>20} | {'fine-tune':>20} | ft - scratch")
    for lr, p, loss in itertools.product(GRID["lr"], GRID["drop_rate"], GRID["loss"]):
        pair = run_pair(replace(MAIN, lr=lr, drop_rate=p, loss=loss))
        s, f = pair["default"].val_at(), pair["1990"].val_at()
        rows.append({"lr": lr, "drop_rate": p, "loss": loss,
                     "scratch_folds": s.tolist(), "finetune_folds": f.tolist()})
        print(f"{lr:>7.0e} {p:>5.1f} {loss:>4} | {format_mae(s.mean(), s.std(ddof=1)):>20} | "
              f"{format_mae(f.mean(), f.std(ddof=1)):>20} | {f.mean() - s.mean():>+10,.0f}", flush=True)
    import json
    (RESULTS_DIR / f"{NAME}_sweep.json").write_text(json.dumps(
        {"base": MAIN.to_dict(), "seed_stride": STRIDE, "rows": rows}, indent=1))
    worse = sum(np.mean(r["finetune_folds"]) > np.mean(r["scratch_folds"]) for r in rows)
    print(f"fine-tune ends worse in {worse} of {len(rows)} settings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="main settings, 2000 epochs")
    ap.add_argument("--sweep", action="store_true", help="lr x dropout x loss grid")
    a = ap.parse_args()
    configure_runtime()  # one thread, deterministic algorithms: as Task 1A
    if a.sweep:
        sweep()
    elif a.long:
        main_comparison(replace(MAIN, epochs=2000), "_long")
    else:
        main_comparison(MAIN, "")
