"""Task 2D: ablation study with five-fold cross-validation, then (separately, once) the final test.

Every row is trained with the same configuration as the Task 1A MLP and 2C
(configs/task1a_baseline.json): lr 0.001, 500 full-batch Adam steps, dropout 0.1 on the MLP hidden
units, L1 loss, seed 42 + fold (fold_seed_stride 1), a fresh model and fresh standardisation per
fold, CV MAE at the final epoch, mean +/- sample s.d. (ddof 1). Same folds and seeds in every row,
so fold-by-fold differences are reported as well.

Rows: the Task 1A MLP, and the cross-attention network with every combination of three components
(8 rows), so any "remove one component at a time" path down to the MLP can be read off the table:

    LoRA            rank 4 on W_Q, W_K, W_V                  make_model_fn(lora_rank=4)
    weight dropout  0.1 on the entries of W_Q, W_K, W_V      make_model_fn(weight_drop=0.1)
    transfer        MLP part initialised from the 1990 net    Config(init="1990")

0.1 is the group's dropout rate (Task 1A, 2A4, 2B2). The cross-attention row with no components is
2C's baseline row, retrained here so that every row comes from the same machine.

    python -m experiments.t2d_ablation                     (CV, ~50 min with 2 processes)
    python -m experiments.t2d_ablation --from-saved        (table + figure only)
    python -m experiments.t2d_ablation --epochs 5          (smoke test, writes t2d_ablation_smoke.*)
    python -m experiments.t2d_ablation --final-test LABEL  (ONCE, after the group fixes the best row:
                                                            trains LABEL on all 500 examples and
                                                            evaluates on the test set)

Writes results/t2d_ablation.json, .csv, .png; the final test writes t2d_final_test.json, a lock
file t2d_locked_config.json and the model t2d_model.pt.
"""
import argparse
import csv
import json
from dataclasses import replace
from itertools import product
from multiprocessing import Pool

import numpy as np
import torch

from housing.attention_extras import make_model_fn
from housing.data import N_FOLDS, all_train, fold_split, load_test
from housing.mlp import count_parameters
from housing.report import RESULTS, cv_result, fit_result, plot_curves
from housing.train import (CVResult, Config, History, configure_runtime, default_model_fn, mae_dollars,
                           train_full_batch)

NAME: str = "t2d_ablation"
CONFIG: Config = Config(lr=1e-3, epochs=500, drop_rate=0.1, loss="l1", seed=42)
FOLD_SEED_STRIDE: int = 1
DDOF: int = 1
RECORD_EVERY: int = 10
LORA_RANK: int = 4
WEIGHT_DROP: float = 0.1
MLP_ROW: str = "Task 1A MLP"
EVAL_BATCH: int = 2000  # the test set needs batches: all 21,282 rows at once is a 1.76 GB attention matrix


def row(lora: bool, weight_drop: bool, transfer: bool) -> dict:
    parts = [name for on, name in ((lora, "LoRA"), (weight_drop, "weight dropout"), (transfer, "transfer")) if on]
    return {"label": "cross-attention" + "".join(f" + {p}" for p in parts), "attention": True,
            "lora": lora, "weight_dropout": weight_drop, "transfer": transfer}


ROWS: list[dict] = [{"label": MLP_ROW, "attention": False, "lora": False, "weight_dropout": False,
                     "transfer": False}] + [row(*c) for c in product((False, True), repeat=3)]
LABELS: list[str] = [r["label"] for r in ROWS]

# The ablation path for the report table and figure: the best cross-attention combination by CV,
# then one component removed at a time (the smaller contribution first) down to the Task 1A MLP.
# Chosen after the CV runs (results/t2d_ablation.json); the full 9-row table is in the CSV.
ABLATION_PATH: list[str] = ["cross-attention + LoRA + weight dropout", "cross-attention + weight dropout",
                            "cross-attention", MLP_ROW]


def model_fn_and_config(r: dict, cfg: Config):
    """The model function and Config of a row (transfer changes only Config.init)."""
    cfg = replace(cfg, init="1990" if r["transfer"] else "default")
    if not r["attention"]:
        return default_model_fn, cfg
    options = {}
    if r["lora"]:
        options["lora_rank"] = LORA_RANK
    if r["weight_dropout"]:
        options["weight_drop"] = WEIGHT_DROP
    return make_model_fn(**options), cfg


def run_fold(job: tuple[str, int, int]) -> tuple[str, int, History, int]:
    """One CV round, as cross_validate() runs it: fresh model and statistics, seed + k * stride."""
    label, k, epochs = job
    configure_runtime()
    train, val = fold_split(k)
    model_fn, cfg = model_fn_and_config(ROWS[LABELS.index(label)], replace(CONFIG, epochs=epochs))
    cfg = replace(cfg, seed=cfg.seed + k * FOLD_SEED_STRIDE)
    fit = train_full_batch(cfg, train, val, model_fn, record_every=RECORD_EVERY)
    return label, k, fit.history, count_parameters(fit.model)


def summarise(results: dict[str, dict]) -> list[dict]:
    """Per row: CV MAE at the final epoch, and the fold-by-fold difference from the Task 1A MLP and
    from plain cross-attention (same folds and seeds, so pairing removes fold-to-fold difficulty)."""
    refs = {name: np.array(results[name]["fold_mae"]) for name in (MLP_ROW, "cross-attention")}
    out = []
    for r in ROWS:
        res = results[r["label"]]
        folds = np.array(res["fold_mae"])
        entry = {**r, "trainable_parameters": res["trainable_parameters"], "cv_mae": res["mean_mae"],
                 "cv_sd": res["std_mae"], "best_mean_epoch": res["best_mean_epoch"],
                 "best_mean_validation_mae": res["best_mean_validation_mae"]}
        for name, key in ((MLP_ROW, "mlp"), ("cross-attention", "attention")):
            diff = folds - refs[name]
            entry[f"mean_diff_vs_{key}"] = float(diff.mean())
            entry[f"sd_diff_vs_{key}"] = float(diff.std(ddof=DDOF))
            entry[f"folds_better_than_{key}"] = int((diff < 0).sum())
        entry.update({f"fold_{k}": float(v) for k, v in enumerate(folds)})
        out.append(entry)
    return out


def report(results: dict[str, dict], name: str, cfg: Config) -> None:
    summary = summarise(results)
    print(f"{'row':<52} {'params':>6} | {'CV MAE (final epoch)':>22} | {'vs MLP':>17} | {'vs attention':>17}")
    for s in sorted(summary, key=lambda s: s["cv_mae"]):
        print(f"{s['label']:<52} {s['trainable_parameters']:>6,} | {s['cv_mae']:>10,.0f} +/- {s['cv_sd']:>7,.0f} | "
              f"{s['mean_diff_vs_mlp']:>+9,.0f} ({s['folds_better_than_mlp']}/5) | "
              f"{s['mean_diff_vs_attention']:>+9,.0f} ({s['folds_better_than_attention']}/5)", flush=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"task": "2D", "config": cfg.to_dict(), "fold_seed_stride": FOLD_SEED_STRIDE,
                                "std_ddof": DDOF, "record_every": RECORD_EVERY, "lora_rank": LORA_RANK,
                                "weight_drop": WEIGHT_DROP, "runs": results, "summary": summary},
                               indent=2, allow_nan=False) + "\n")
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    cvs = {}
    for label in ABLATION_PATH:  # all 9 rows in one figure are unreadable; the table has them all
        hs = [History(epoch=f["curves"]["epoch"], train_mae=f["curves"]["train_mae"],
                      val_mae=f["curves"]["validation_mae"]) for f in results[label]["folds"]]
        cvs[label] = CVResult(cfg, hs)
    plot_curves(cvs, RESULTS / f"{name}.png",
                title=f"2D ablation path: five-fold CV, lr {cfg.lr:g}, {cfg.epochs} epochs, hidden dropout "
                      f"{cfg.drop_rate:g}, {cfg.loss.upper()} (mean, band: s.d. over folds)")
    print(path)


def final_test(label: str) -> None:
    """Train the chosen row on all 500 examples and evaluate it once on the test set.
    The lock file is created exclusively before training, so the test set is reached at most once."""
    result_path, lock = RESULTS / "t2d_final_test.json", RESULTS / "t2d_locked_config.json"
    if result_path.exists():
        print(f"Existing final evaluation: test MAE ${json.loads(result_path.read_text())['test_mae']:,.2f}; "
              "no retraining or test access.")
        return
    saved = json.loads((RESULTS / f"{NAME}.json").read_text())
    cv = next(s for s in saved["summary"] if s["label"] == label)
    r = ROWS[LABELS.index(label)]
    model_fn, cfg = model_fn_and_config(r, CONFIG)
    with lock.open("x") as f:  # fails if a lock exists: no repeated or concurrent final tests
        json.dump({"row": label, "config": cfg.to_dict(), "components": r, "selected_using":
                   "five-fold CV MAE at the final epoch (results/t2d_ablation.json)",
                   "cv_mae": cv["cv_mae"], "cv_sd": cv["cv_sd"]}, f, indent=2)
    configure_runtime()
    rows = all_train()
    fit = train_full_batch(cfg, rows, None, model_fn, record_every=RECORD_EVERY)
    training = fit_result(cfg, fit, len(rows))
    if training["status"] != "completed":
        raise RuntimeError(training["status"])
    torch.save({"state_dict": fit.model.state_dict(), "config": cfg.to_dict(), "row": r,
                "scaler": fit.std.to_dict()}, RESULTS / "t2d_model.pt")
    test_rows = load_test()  # the only test-set access in Task 2
    test_mae = mae_dollars(fit.model, fit.std, test_rows, batch=EVAL_BATCH)
    t1a = json.loads((RESULTS / "t1a5_final_test.json").read_text())["test_mae"]
    result_path.write_text(json.dumps({"task": "2D", "row": label, "components": r, "config": cfg.to_dict(),
                                       "cv_mae": cv["cv_mae"], "cv_sd": cv["cv_sd"], "test_mae": test_mae,
                                       "n_test": len(test_rows), "task1a_test_mae": t1a, "training": training,
                                       "checkpoint": "t2d_model.pt"}, indent=2, allow_nan=False) + "\n")
    print(f"{label}: test MAE ${test_mae:,.2f} (Task 1A: ${t1a:,.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D: ablation with five-fold CV; final test once")
    parser.add_argument("--from-saved", action="store_true", help="rebuild table and figure, no training")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs, help="anything else is a smoke test")
    parser.add_argument("--processes", type=int, default=2, help="each attention worker holds a 400 x 20,640 "
                        "attention matrix and its gradients; 2 fit an 8 GB laptop")
    parser.add_argument("--final-test", metavar="LABEL", choices=LABELS,
                        help="train this row on all 500 examples and evaluate ONCE on the test set")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.final_test:
        final_test(args.final_test)
        return
    name = NAME if args.epochs == CONFIG.epochs else f"{NAME}_smoke"
    cfg = replace(CONFIG, epochs=args.epochs)
    if args.from_saved:
        results = json.loads((RESULTS / f"{name}.json").read_text())["runs"]
    else:
        # attention rows first (slow), the tiny MLP last, so the two workers stay busy
        jobs = [(label, k, args.epochs) for label in LABELS[1:] + LABELS[:1] for k in range(N_FOLDS)]
        with Pool(args.processes) as pool:
            done = pool.map(run_fold, jobs, chunksize=1)
        results = {}
        for label in LABELS:
            folds = sorted((d for d in done if d[0] == label), key=lambda d: d[1])
            cv = CVResult(cfg, [d[2] for d in folds])
            res = cv_result(cv, ddof=DDOF)
            res.update(trainable_parameters=folds[0][3], best_mean_epoch=cv.best_epoch(),
                       best_mean_validation_mae=float(cv.mean_curve("val_mae")[1].min()))
            results[label] = res
    report(results, name, cfg)


if __name__ == "__main__":
    main()
