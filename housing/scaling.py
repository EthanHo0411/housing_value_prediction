"""Standardisation (spec, section 2). The statistics are part of a trained model.

Fit a Standardiser on the training rows only (the four training folds in a CV round, or all
500 rows for a final model) and send every example that goes through that model, training,
validation or test, through the same statistics.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from housing.data import DATA_DIR, FEATURES, TARGET, Rows

# Population standard deviation (divide by N), the convention mlp_1990.json uses.
DDOF: int = 0


@dataclass(frozen=True)
class Standardiser:
    mu_x: np.ndarray  # (6,)
    sd_x: np.ndarray  # (6,)
    mu_y: float
    sd_y: float

    @classmethod
    def fit(cls, rows: Rows, ddof: int = DDOF) -> "Standardiser":
        sd_x = rows.x.std(axis=0, ddof=ddof)
        sd_y = float(rows.y.std(ddof=ddof))
        if np.any(sd_x == 0) or sd_y == 0:
            raise ValueError("a column has zero standard deviation on these rows")
        return cls(rows.x.mean(axis=0), sd_x, float(rows.y.mean()), sd_y)

    @classmethod
    def from_json(cls, path: Path = DATA_DIR / "mlp_1990.json") -> "Standardiser":
        """The 1990 statistics the provided network was trained with."""
        meta = json.loads(Path(path).read_text())
        names = tuple(c["name"] for c in meta["inputs"])
        if names != FEATURES or meta["price"]["name"] != TARGET:
            raise ValueError(f"{path} columns {names} do not match {FEATURES}")
        return cls(np.array([c["mean"] for c in meta["inputs"]]), np.array([c["std"] for c in meta["inputs"]]),
                   float(meta["price"]["mean"]), float(meta["price"]["std"]))

    def x(self, x_raw: np.ndarray) -> torch.Tensor:
        """Raw inputs (N, 6) -> standardised float32 tensor (N, 6)."""
        return torch.as_tensor((x_raw - self.mu_x) / self.sd_x, dtype=torch.float32)

    def y(self, y_raw: np.ndarray) -> torch.Tensor:
        """Prices in dollars (N,) -> standardised float32 tensor (N, 1), the network's target."""
        return torch.as_tensor((y_raw - self.mu_y) / self.sd_y, dtype=torch.float32).reshape(-1, 1)

    def dollars(self, f: torch.Tensor) -> np.ndarray:
        """Network output (N, 1), standardised -> prices in dollars (N,): sigma_y * f(x) + mu_y."""
        return self.sd_y * f.detach().double().reshape(-1).numpy() + self.mu_y

    def to_dict(self):
        return {"mu_x": self.mu_x.tolist(), "sd_x": self.sd_x.tolist(),
                "mu_y": self.mu_y, "sd_y": self.sd_y}

    @classmethod
    def from_dict(cls, values):
        """Restore statistics, including those in historical checkpoints."""
        if "mu_x" in values:
            return cls(np.array(values["mu_x"]), np.array(values["sd_x"]),
                       values["mu_y"], values["sd_y"])
        return cls(np.array(values["x_mean"]), np.array(values["x_std"]),
                   values["y_mean"], values["y_std"])
