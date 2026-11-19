"""Load the coursework data with a fixed column order, and the given cross-validation folds.

Only the 500 rows of housing_2020_train.csv are for training and validation (spec, section 2).
The test set is reached through load_test() alone, which only the two final evaluations
(Task 1A5 and Task 2D) should call.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

# The order of mlp_1990.json's inputs: the 1990 network expects exactly this order.
FEATURES: tuple[str, ...] = ("median_income", "population", "households", "housing_age", "latitude", "longitude")
TARGET: str = "median_house_value"
N_FOLDS: int = 5


@dataclass(frozen=True)
class Rows:
    """Raw (unstandardised) examples: x is (N, 6) in FEATURES order, y is (N,) in dollars."""

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape[1] != len(FEATURES):
            raise ValueError(f"x must be (N, {len(FEATURES)}), got {self.x.shape}")
        if self.y.shape != (self.x.shape[0],):
            raise ValueError(f"y must be ({self.x.shape[0]},), got {self.y.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "Rows":
        return Rows(self.x[idx], self.y[idx])


def _read(name: str, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    missing = [c for c in (*FEATURES, TARGET, *extra) if c not in df.columns]
    if missing:
        raise ValueError(f"{name} lacks columns {missing}")
    if df[[*FEATURES, TARGET]].isna().any().any():
        raise ValueError(f"{name} has missing values")
    return df


def _rows(df: pd.DataFrame) -> Rows:
    x = df[list(FEATURES)].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)
    x.setflags(write=False)  # shared and cached: nobody may edit them in place
    y.setflags(write=False)
    return Rows(x, y)


@lru_cache(maxsize=None)
def load_train() -> tuple[Rows, np.ndarray]:
    """The 500 training examples from 2020 and their given fold labels (0 to 4)."""
    df = _read("housing_2020_train.csv", extra=("fold",))
    folds = df["fold"].to_numpy(dtype=np.int64)
    folds.setflags(write=False)
    return _rows(df), folds


@lru_cache(maxsize=None)
def load_1990() -> Rows:
    """The 20,640 records from 1990 (Task 2's source tokens)."""
    return _rows(_read("housing_1990.csv"))


def load_test() -> Rows:
    """The 21,282 test examples from 2020. Call only for the final evaluations (1A5, 2D)."""
    return _rows(_read("housing_2020_test.csv"))


def fold_split(k: int) -> tuple[Rows, Rows]:
    """Cross-validation round k: training rows = the other four folds, validation rows = fold k."""
    if k not in range(N_FOLDS):
        raise ValueError(f"fold must be in 0..{N_FOLDS - 1}, got {k}")
    rows, folds = load_train()
    return rows.subset(np.flatnonzero(folds != k)), rows.subset(np.flatnonzero(folds == k))


def all_train() -> Rows:
    """All 500 training rows, for the final models (1A5, 2D)."""
    return load_train()[0]
