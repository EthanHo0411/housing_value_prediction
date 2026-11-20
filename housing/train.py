"""Full-batch training with MAE curves, five-fold cross-validation, and batched evaluation.

Protocol (spec, section 2): full-batch Adam, so one epoch is one gradient step; a fixed seed;
training and validation MAE recorded in dollars, with dropout switched off while measuring;
a fresh model and fresh standardisation statistics in every cross-validation round.
"""
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
from torch import nn

from housing.data import N_FOLDS, Rows, fold_split
from housing.mlp import MLP, load_mlp_1990
from housing.scaling import Standardiser

LOSSES: dict[str, Callable[[], nn.Module]] = {"mse": nn.MSELoss, "l1": nn.L1Loss}


@dataclass(frozen=True)
class Config:
    """One training setting. Experiments change these fields and nothing else."""

    lr: float
    epochs: int = 2000
    drop_rate: float = 0.0
    loss: str = "mse"       # "mse" or "l1" (Task 1A4), on the standardised price
    seed: int = 0           # the same seed in every fold: every round starts from the same weights
    init: str = "default"   # "default" (PyTorch's initialisation) or "1990" (the provided weights)

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {self.loss!r}")
        if self.init not in ("default", "1990"):
            raise ValueError(f"init must be 'default' or '1990', got {self.init!r}")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ModelFn = Callable[[Config], nn.Module]


def default_model_fn(cfg: Config) -> nn.Module:
    """The Task 1 network: PyTorch's initialisation, or the 1990 weights for fine-tuning."""
    return load_mlp_1990(cfg.drop_rate) if cfg.init == "1990" else MLP(cfg.drop_rate)


@dataclass
class History:
    """Curves of one training run. Entry 0 of epoch, train_mae and val_mae is before the first step."""

    epoch: list[int] = field(default_factory=list)
    train_mae: list[float] = field(default_factory=list)   # dollars, dropout off
    val_mae: list[float] = field(default_factory=list)     # dollars, dropout off; empty without validation rows
    loss: list[float] = field(default_factory=list)        # loss of step e is loss[e - 1] (standardised units)
    diverged_at: int | None = None                          # first step whose loss was not finite


@dataclass
class Fit:
    model: nn.Module
    std: Standardiser
    history: History


def raw_tensor(rows: Rows) -> torch.Tensor:
    """The raw inputs as a float64 tensor, the x_raw argument of every model (a copy: rows stay read-only)."""
    return torch.tensor(rows.x, dtype=torch.float64)


def predict_dollars(model: nn.Module, std: Standardiser, rows: Rows, batch: int | None = None) -> np.ndarray:
    """Predicted prices in dollars (N,), in evaluation mode, optionally in batches of rows.

    Batches keep Task 2's attention matrix small: all 21,282 test rows against 20,640 records
    at once would need 1.76 GB for that matrix alone.
    """
    was_training = model.training
    model.eval()
    x_std, x_raw = std.x(rows.x), raw_tensor(rows)
    step = batch or len(rows)
    with torch.no_grad():
        out = torch.cat([model(x_std[i:i + step], x_raw[i:i + step]) for i in range(0, len(rows), step)])
    model.train(was_training)
    return std.dollars(out)


def mae_dollars(model: nn.Module, std: Standardiser, rows: Rows, batch: int | None = None) -> float:
    return float(np.mean(np.abs(predict_dollars(model, std, rows, batch) - rows.y)))


def train_full_batch(cfg: Config, train_rows: Rows, val_rows: Rows | None = None,
                     model_fn: ModelFn = default_model_fn, std: Standardiser | None = None,
                     record_every: int = 1, eval_batch: int | None = None) -> Fit:
    """Train one fresh model on train_rows; record MAE every record_every epochs (and at the end).

    std: by default fitted on train_rows. Pass one to override (moment matching passes the
    2020 statistics to the 1990 network; epochs=0 then evaluates without training).
    """
    std = std or Standardiser.fit(train_rows)
    torch.manual_seed(cfg.seed)  # before building: the same seed gives the same initial weights
    model = model_fn(cfg)
    x_std, y_std, x_raw = std.x(train_rows.x), std.y(train_rows.y), raw_tensor(train_rows)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.epochs > 0 and not trainable:
        raise ValueError("the model has no trainable parameters; use epochs=0 to evaluate only")
    opt = torch.optim.Adam(trainable, lr=cfg.lr) if cfg.epochs > 0 else None
    loss_fn = LOSSES[cfg.loss]()
    hist = History()

    def record(epoch: int) -> None:
        hist.epoch.append(epoch)
        hist.train_mae.append(mae_dollars(model, std, train_rows, eval_batch))
        if val_rows is not None:
            hist.val_mae.append(mae_dollars(model, std, val_rows, eval_batch))

    record(0)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_std, x_raw), y_std)
        value = loss.item()
        hist.loss.append(value)
        if not math.isfinite(value):  # diverged (too high a learning rate): keep the curve so far
            hist.diverged_at = epoch
            break
        loss.backward()
        opt.step()
        if epoch % record_every == 0 or epoch == cfg.epochs:
            record(epoch)
    model.eval()
    return Fit(model, std, hist)


@dataclass
class CVResult:
    config: Config
    histories: list[History]   # one per fold, fold k at index k

    def val_at(self, epoch: int | None = None) -> np.ndarray:
        """Validation MAE of each fold at a recorded epoch (default: the last recorded one)."""
        if epoch is None:
            return np.array([h.val_mae[-1] for h in self.histories])
        return np.array([h.val_mae[h.epoch.index(epoch)] for h in self.histories])

    def summary(self, epoch: int | None = None, ddof: int = 0) -> tuple[float, float]:
        """CV MAE as (mean, standard deviation) over the folds, in dollars."""
        vals = self.val_at(epoch)
        return float(vals.mean()), float(vals.std(ddof=ddof))

    def mean_curve(self, which: str = "val_mae") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(epochs, mean, sd) of a curve over the folds, on the epochs all folds recorded."""
        n = min(len(getattr(h, which)) for h in self.histories)
        curves = np.array([getattr(h, which)[:n] for h in self.histories])
        return np.array(self.histories[0].epoch[:n]), curves.mean(axis=0), curves.std(axis=0, ddof=0)

    def best_epoch(self) -> int:
        """The recorded epoch with the lowest mean validation MAE: a candidate training budget."""
        epochs, mean, _ = self.mean_curve("val_mae")
        return int(epochs[int(np.argmin(mean))])


def cross_validate(cfg: Config, model_fn: ModelFn = default_model_fn,
                   std_fn: Callable[[Rows], Standardiser] | None = None,
                   record_every: int = 1, eval_batch: int | None = None,
                   fold_seed_stride: int = 0) -> CVResult:
    """Five-fold CV on the given folds: round k trains a fresh model on the other four folds.

    fold_seed_stride defaults to 0 (same seed); Task 1A uses 1 to preserve its recorded runs.
    std_fn builds the statistics from each round's training rows (default: Standardiser.fit).
    """
    histories = []
    for k in range(N_FOLDS):
        train_rows, val_rows = fold_split(k)
        std = std_fn(train_rows) if std_fn else None
        fold_cfg = replace(cfg, seed=cfg.seed + k * fold_seed_stride)
        fit = train_full_batch(fold_cfg, train_rows, val_rows, model_fn, std, record_every, eval_batch)
        histories.append(fit.history)
    return CVResult(cfg, histories)
