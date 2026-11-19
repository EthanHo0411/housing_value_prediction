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
