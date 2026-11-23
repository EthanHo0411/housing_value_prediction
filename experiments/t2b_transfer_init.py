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
results/t2b1_transfer_init_overlay.png.

Task 2B2 (--b2): can dropout and LoRA be used together with the transferred initialisation?
Same base setting as 2B1. Three regularisers, each trained with both initialisations:
    dropout          MLP hidden dropout, Config(drop_rate=0.1): Task 1A's selected rate
    LoRA             rank 4 on W_Q, W_K, W_V (P4's apply_lora, via make_model_fn)
    dropout + LoRA   both
The unregularised pair is 2B1 (read from results/t2b1_transfer_init.json). Within each
regulariser the two runs differ only in the MLP's initial weights: same attention weights,
same LoRA A, same random state, so the same dropout masks. To use the dropout the group picks
in 2A3 instead, pass --drop-rate / --attn-drop / --weight-drop.

    python -m experiments.t2b_transfer_init --b2                 (~30 min with 3 processes)
    python -m experiments.t2b_transfer_init --b2 --epochs 30     (smoke test)
    python -m experiments.t2b_transfer_init --b2 --from-saved    (summary + figure only)

Writes results/t2b2_dropout_lora.json (+ .png, .csv) and results/t2b2_dropout_lora_overlay.png.
"""
import argparse
import json
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.t2a2_overfit import RECORD_EVERY, ZOOM_EPOCHS, run, summarise
from housing.attention import D_K
from housing.attention_extras import LORA_RANK, make_model_fn
from housing.data import fold_split
from housing.mlp import count_parameters
from housing.report import RESULTS, fit_result, save_result
from housing.train import Config, configure_runtime, train_full_batch

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


# ---------------------------------------------------------------- 2B2: + dropout, LoRA

NAME_B2: str = "t2b2_dropout_lora"


def arms(drop_rate: float, attn_drop: float, weight_drop: float) -> dict[str, dict]:
    """Regulariser name -> (Config fields, model options). 'none' is 2B1's pair."""
    extra = {k: v for k, v in (("attn_drop", attn_drop), ("weight_drop", weight_drop)) if v}
    return {
        "none": {"drop_rate": 0.0, "options": {}},
        "dropout": {"drop_rate": drop_rate, "options": dict(extra)},
        "LoRA": {"drop_rate": 0.0, "options": {"lora_rank": LORA_RANK}},
        "dropout + LoRA": {"drop_rate": drop_rate, "options": {"lora_rank": LORA_RANK, **extra}},
    }


def run_variant(job: tuple[str, Config, dict]) -> dict:
    """2A2's run() with a model function from attention_extras; same recording and saving."""
    arm, cfg, options = job
    configure_runtime()
    train, val = fold_split(0)
    model_fn = make_model_fn(**options)
    fit = train_full_batch(cfg, train, val, model_fn, record_every=RECORD_EVERY)
    h = fit.history
    best = min(range(len(h.val_mae)), key=h.val_mae.__getitem__)
    result = fit_result(cfg, fit, len(train), len(val))
    result["best_validation"] = {"epoch": h.epoch[best], "validation_mae": h.val_mae[best],
                                 "train_mae": h.train_mae[best]}
    result["arm"] = arm
    result["model_options"] = options
    result["trainable_parameters"] = count_parameters(fit.model)
    return result


def overlay_b2(runs: list[dict], path) -> None:
    """One panel per regulariser: both initialisations, validation solid, training dashed."""
    names = list(dict.fromkeys(r["arm"] for r in runs))
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.0), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for r in (r for r in runs if r["arm"] == name):
            c = "C0" if r["config"]["init"] == "default" else "C1"
            label = LABELS[r["config"]["init"]].replace(" (2A2 baseline)", "")
            cur = r["curves"]
            ax.plot(cur["epoch"], cur["validation_mae"], "-", color=c, lw=1.3, label=f"{label} (validation)")
            ax.plot(cur["epoch"], cur["train_mae"], "--", color=c, lw=0.9, label=f"{label} (training)")
            b = r["best_validation"]
            ax.plot(b["epoch"], b["validation_mae"], "o", color=c, ms=4)
        ax.set(title=name, xlabel="epoch", yscale="log")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("MAE ($)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    cfg = runs[0]["config"]
    fig.suptitle(f"2B2: transferred vs default MLP initialisation under each regulariser "
                 f"(lr {cfg['lr']:g}, {cfg['loss'].upper()} loss, validation = fold 0; dots: best validation)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_b2(runs: list[dict], settings: dict) -> None:
    summary = summarise(runs)
    for s, r in zip(summary, runs):
        s.update(arm=r["arm"], init=r["config"]["init"], trainable_parameters=r["trainable_parameters"])
    print(f"{'regulariser':>15} {'init':>8} {'params':>7} | {'best val':>10} {'epoch':>6} | "
          f"{'final train':>11} {'mean last 10':>12} {'final val':>10}")
    for s in summary:
        print(f"{s['arm']:>15} {s['init']:>8} {s['trainable_parameters']:>7,} | {s['best_validation_mae']:>10,.0f} "
              f"{s['best_epoch']:>6} | {s['final_train_mae']:>11,.0f} {s['mean_train_mae_last_10_records']:>12,.0f} "
              f"{s['final_validation_mae']:>10,.0f}   ({s['status']})")
    print("\ntransferred - default (negative: the 1990 weights help)")
    for name in dict.fromkeys(s["arm"] for s in summary):
        d, t = (next(s for s in summary if s["arm"] == name and s["init"] == i) for i in ("default", "1990"))
        print(f"{name:>15}: best val {t['best_validation_mae'] - d['best_validation_mae']:>+9,.0f}   "
              f"final val {t['final_validation_mae'] - d['final_validation_mae']:>+9,.0f}")
    path = save_result(NAME_B2, {"task": "2B2", "validation_fold": 0, "settings": settings,
                                 "model": {"name": "AttentionVariant", "d_k": D_K, "record_every": RECORD_EVERY,
                                           "unregularised_pair": NAME},
                                 "runs": runs, "summary": summary})
    overlay_b2(runs, RESULTS / f"{NAME_B2}_overlay.png")
    print(path)


def main_b2(a: argparse.Namespace) -> None:
    settings = {"drop_rate": a.drop_rate, "attn_drop": a.attn_drop, "weight_drop": a.weight_drop,
                "lora_rank": LORA_RANK}
    if a.from_saved:
        runs = json.loads((RESULTS / f"{NAME_B2}.json").read_text())["runs"]
        report_b2(runs, settings)
        return
    jobs = []
    for name, arm in arms(a.drop_rate, a.attn_drop, a.weight_drop).items():
        if name == "none":
            continue
        for init in ("default", "1990"):
            cfg = Config(lr=a.lr, epochs=a.epochs, loss=a.loss, seed=a.seed, init=init, drop_rate=arm["drop_rate"])
            jobs.append((name, cfg, arm["options"]))
    with Pool(a.processes) as pool:
        runs = pool.map(run_variant, jobs, chunksize=1)

    # The unregularised pair: 2B1's runs if they match these settings, otherwise train them too.
    base = [Config(lr=a.lr, epochs=a.epochs, loss=a.loss, seed=a.seed, init=i).to_dict() for i in ("default", "1990")]
    path = RESULTS / f"{NAME}.json"
    saved = json.loads(path.read_text())["runs"] if path.exists() else []
    pair = [next((r for r in saved if r["config"] == c), None) for c in base]
    if any(r is None for r in pair):
        print("2B1 runs with these settings not found: training the unregularised pair as well")
        with Pool(2) as pool:
            pair = pool.map(run_variant, [("none", Config(**c), {}) for c in base], chunksize=1)
    else:
        print(f"unregularised pair read from {path.name}")
        pair = [{**r, "arm": "none", "model_options": {}, "trainable_parameters": 17089} for r in pair]
    report_b2(pair + runs, settings)


def main() -> None:
    p = argparse.ArgumentParser(description="2B: transferred initialisation of the MLP part")
    p.add_argument("--b2", action="store_true", help="2B2: with dropout and LoRA")
    p.add_argument("--lr", type=float, default=1e-3, help="the group's Task 2 base rate (TASK_2A.md)")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--loss", default="l1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-rate", type=float, default=0.1, help="2B2: MLP hidden dropout (Task 1A: 0.1)")
    p.add_argument("--attn-drop", type=float, default=0.0, help="2B2: dropout on attention weights")
    p.add_argument("--weight-drop", type=float, default=0.0, help="2B2: dropout on entries of W_Q, W_K, W_V")
    p.add_argument("--processes", type=int, default=3, help="2B2: parallel runs (6 in total)")
    p.add_argument("--from-saved", action="store_true", help="rebuild summary and figure, no training")
    a = p.parse_args()

    if a.b2:
        main_b2(a)
        return
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