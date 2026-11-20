"""Save runs to results/, load them back, and turn them into the report's figures and tables.

Experiments save; figures and tables are made from the saved files, never by retraining,
so every number in the report maps to one results/<name>.json and the command that wrote it.
"""
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; no window (works on lab machines without a display)
import matplotlib.pyplot as plt
import numpy as np

from housing.train import Config, CVResult, History

RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "results"


def save_run(name: str, cfg: Config, histories: list[History], note: str = "") -> Path:
    """Write the config and the curves of one run (one history) or one CV (five) to results/<name>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"name": name, "note": note, "config": cfg.to_dict(),
                                "histories": [asdict(h) for h in histories]}, indent=1))
    return path


def load_run(name: str) -> CVResult:
    """A saved run as a CVResult (with one history for a single train/validation run)."""
    data = json.loads((RESULTS_DIR / f"{name}.json").read_text())
    return CVResult(Config(**data["config"]), [History(**h) for h in data["histories"]])


def format_mae(mean: float, sd: float | None = None) -> str:
    """'52,416' or '52,416 +/- 1,234' (dollars, rounded)."""
    return f"{mean:,.0f}" if sd is None else f"{mean:,.0f} +/- {sd:,.0f}"


def summary_table(runs: dict[str, CVResult], epoch: int | None = None, latex: bool = False) -> str:
    """One row per run: label and CV MAE (mean +/- s.d. over the folds) at the given epoch."""
    lines = []
    for label, run in runs.items():
        mean, sd = run.summary(epoch)
        cell = f"{mean:,.0f} $\\pm$ {sd:,.0f}" if latex else format_mae(mean, sd)
        lines.append(f"{label} & {cell} \\\\" if latex else f"| {label} | {cell} |")
    header = "" if latex else "| Run | CV MAE ($), mean +/- s.d. |\n|---|---|\n"
    return header + "\n".join(lines)


def plot_curves(runs: dict[str, CVResult], path: Path | str, title: str = "",
                show_train: bool = True, log_y: bool = True) -> Path:
    """Training (dashed) and validation (solid) MAE against epoch, one colour per run.

    A run with several folds is drawn as its mean curve with a +/- s.d. band.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (label, run) in enumerate(runs.items()):
        colour = f"C{i}"
        for which, style in (("val_mae", "-"), ("train_mae", "--")):
            if which == "train_mae" and not show_train:
                continue
            epochs, mean, sd = run.mean_curve(which)
            if len(epochs) == 0:
                continue
            name = f"{label} ({'validation' if which == 'val_mae' else 'training'})"
            ax.plot(epochs, mean, style, color=colour, label=name, linewidth=1.4)
            if len(run.histories) > 1:
                ax.fill_between(epochs, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
    ax.set(xlabel="epoch (one full-batch step)", ylabel="MAE ($)", title=title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# Task 1A JSON format and figure regeneration (also reads pre-merge results).
import argparse
import csv
RESULTS = RESULTS_DIR

def save_result(name, result):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    render(path)
    return path


def render(path):
    path = Path(path)
    result = json.loads(path.read_text())
    runs = result.get("runs", [])
    rows = []
    if not runs:
        if "test_mae" in result:
            rows = [{"loss": result["config"]["loss"], "dropout": result["config"]["dropout"],
                     "cv_mae": result["selected_cv_mae"], "test_mae": result["test_mae"]}]
    else:
        fig, axes = plt.subplots(len(runs), 1, figsize=(8, 2.8 * len(runs)), squeeze=False)
        for ax, run in zip(axes[:, 0], runs):
            cfg = run["config"]
            title = f"lr={cfg['learning_rate']:g}, dropout={cfg['dropout']:g}, loss={cfg['loss']}"
            if "folds" in run:
                epochs = run["folds"][0]["curves"]["epoch"]
                for key, label in [("train_mae", "Training"), ("validation_mae", "Validation")]:
                    values = np.array([f["curves"][key] for f in run["folds"]])
                    mean, std = values.mean(0), values.std(0, ddof=1)
                    line, = ax.plot(epochs, mean, label=label)
                    ax.fill_between(epochs, mean - std, mean + std, alpha=.15, color=line.get_color())
                rows.append({"dropout": cfg["dropout"], "loss": cfg["loss"], "mean_mae": run["mean_mae"],
                             "std_mae": run["std_mae"], **{f"fold_{k}": v for k, v in enumerate(run["fold_mae"])}})
            else:
                curves = run["curves"]
                ax.plot(curves["epoch"], curves["train_mae"], label="Training")
                ax.plot(curves["epoch"], curves["validation_mae"], label="Validation")
                rows.append({"learning_rate": cfg["learning_rate"], "dropout": cfg["dropout"],
                             "train_mae": curves["train_mae"][-1], "validation_mae": curves["validation_mae"][-1],
                             "status": run["status"]})
                if result["task"] == "1A1":
                    ax.set_yscale("log")
            ax.set(title=title, xlabel="Epoch (one full-batch Adam step)", ylabel="MAE ($)")
            ax.grid(alpha=.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args().paths:
        render(path)
