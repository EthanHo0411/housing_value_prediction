"""Task 2C, option C1: position-encode the price (C1.i) and layer normalisation (C1.ii), with
five-fold cross-validation.

Every variant is trained with the Task 1A baseline configuration (configs/task1a_baseline.json):
lr 0.001, 500 full-batch Adam steps, dropout 0.1 on the MLP hidden units, L1 loss, seed 42 + fold
(fold_seed_stride 1), CV MAE at the final epoch, mean +/- sample s.d. (ddof 1) over the five folds.
The same settings as the Task 1A MLP row, so the baseline here is also 2D's "cross-attention" row.
The variants come from P4's housing/attention_extras.py:

    cross-attention   the 2A network (17,089 parameters)
    C1.i              psi(ybar): 32 frequencies 0.5-32, source token 128-dimensional (25,153)
    C1.ii             r = LN(h2) + LN(phi), z = [LN(g2) + LN(phi) ; ybar] (17,601)

    python -m experiments.t2c_variations                (~20 min, 6 processes)
    python -m experiments.t2c_variations --from-saved   (table + figure only)

Writes results/t2c_variations.json, results/t2c_variations.csv and results/t2c_variations.png.
"""
import argparse
import csv
import json
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from housing.attention_extras import make_model_fn
from housing.data import N_FOLDS, fold_split
from housing.mlp import count_parameters
from housing.report import RESULTS, cv_result, plot_curves
from housing.train import CVResult, Config, History, configure_runtime, train_full_batch

NAME: str = "t2c_variations"
CONFIG: Config = Config(lr=1e-3, epochs=500, drop_rate=0.1, loss="l1", seed=42)
FOLD_SEED_STRIDE: int = 1
DDOF: int = 1
RECORD_EVERY: int = 10

VARIANTS: dict[str, dict] = {
    "cross-attention": {},
    "C1.i price encoding": {"price_encoding": True},
    "C1.ii layer norm": {"layer_norm": True},
}
BASELINE: str = "cross-attention"


def run_fold(job: tuple[str, int]) -> tuple[str, int, History, int]:
    """One CV round, as cross_validate() runs it: fresh model and statistics, seed + k * stride."""
    label, k = job
    configure_runtime()
    train, val = fold_split(k)
    cfg = replace(CONFIG, seed=CONFIG.seed + k * FOLD_SEED_STRIDE)
    fit = train_full_batch(cfg, train, val, make_model_fn(**VARIANTS[label]), record_every=RECORD_EVERY)
    return label, k, fit.history, count_parameters(fit.model)


def summarise(results: dict[str, dict]) -> list[dict]:
    """Per variant: CV MAE (final epoch), best epoch of the mean curve, and the fold-by-fold
    difference from the baseline (same folds and seeds, so the pairing removes fold difficulty)."""
    base = np.array(results[BASELINE]["fold_mae"])
    rows = []
    for label, r in results.items():
        folds = np.array(r["fold_mae"])
        diff = folds - base
        rows.append({"variant": label, "trainable_parameters": r["trainable_parameters"],
                     "cv_mae": r["mean_mae"], "cv_sd": r["std_mae"],
                     "best_mean_epoch": r["best_mean_epoch"], "best_mean_validation_mae": r["best_mean_validation_mae"],
                     "mean_diff_vs_baseline": float(diff.mean()),
                     "sd_diff_vs_baseline": float(diff.std(ddof=DDOF)),
                     "folds_better_than_baseline": int((diff < 0).sum()),
                     **{f"fold_{k}": float(v) for k, v in enumerate(folds)}})
    return rows


def report(results: dict[str, dict]) -> None:
    summary = summarise(results)
    print(f"{'variant':<20} {'params':>7} | {'CV MAE (final)':>20} | {'best mean (epoch)':>18} | "
          f"{'diff vs base':>16} better")
    for s in summary:
        print(f"{s['variant']:<20} {s['trainable_parameters']:>7,} | {s['cv_mae']:>10,.0f} +/- {s['cv_sd']:>6,.0f} | "
              f"{s['best_mean_validation_mae']:>10,.0f} ({s['best_mean_epoch']:>4}) | "
              f"{s['mean_diff_vs_baseline']:>+8,.0f} +/- {s['sd_diff_vs_baseline']:>5,.0f} "
              f"{s['folds_better_than_baseline']}/{N_FOLDS}", flush=True)
    path = RESULTS / f"{NAME}.json"
    path.write_text(json.dumps({"task": "2C", "option": "C1", "config": CONFIG.to_dict(),
                                "fold_seed_stride": FOLD_SEED_STRIDE, "std_ddof": DDOF, "record_every": RECORD_EVERY,
                                "runs": results, "summary": summary}, indent=2, allow_nan=False) + "\n")
    with open(RESULTS / f"{NAME}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(path)


def figure(results: dict[str, dict], cfg: Config) -> None:
    """Mean training (dashed) and validation (solid) MAE over the five folds. The band is
    report.plot_curves()'s population s.d. (ddof 0); the table uses the sample s.d. (ddof 1)."""
    cvs = {}
    for label, r in results.items():
        histories = []
        for f in r["folds"]:
            c = f["curves"]
            histories.append(History(epoch=c["epoch"], train_mae=c["train_mae"], val_mae=c["validation_mae"]))
        cvs[label] = CVResult(cfg, histories)
    plot_curves(cvs, RESULTS / f"{NAME}.png",
                title=f"2C (C1): five-fold CV, lr {cfg.lr:g}, {cfg.epochs} epochs, dropout {cfg.drop_rate:g}, "
                      f"{cfg.loss.upper()} (mean, band: s.d. over folds)")


def main() -> None:
    parser = argparse.ArgumentParser(description="2C: C1.i price encoding and C1.ii layer norm, five-fold CV")
    parser.add_argument("--from-saved", action="store_true", help="rebuild table and figure, no training")
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args()
    cfg = CONFIG
    if args.from_saved:
        saved = json.loads((RESULTS / f"{NAME}.json").read_text())
        results, cfg = saved["runs"], Config.from_dict(saved["config"])
    else:
        jobs = [(label, k) for label in VARIANTS for k in range(N_FOLDS)]
        with Pool(args.processes) as pool:
            done = pool.map(run_fold, jobs, chunksize=1)
        results = {}
        for label in VARIANTS:
            folds = sorted((d for d in done if d[0] == label), key=lambda d: d[1])
            cv = CVResult(CONFIG, [d[2] for d in folds])
            r = cv_result(cv, ddof=DDOF)
            r["trainable_parameters"] = folds[0][3]
            r["best_mean_epoch"] = cv.best_epoch()
            r["best_mean_validation_mae"] = float(cv.mean_curve("val_mae")[1].min())
            results[label] = r
    report(results)
    figure(results, cfg)


if __name__ == "__main__":
    main()
