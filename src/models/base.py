"""Common interface for every forecaster.

A forecaster predicts the next-day log return for chosen rows of a Dataset.
Training, the walk-forward backtest and live serving all go through fit/predict,
so a backtested model is exactly the model that is served.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np

from src.models.dataset import Dataset


class Forecaster(ABC):
    name: ClassVar[str]

    def __init__(self, params: dict[str, Any] | None = None, seq_len: int = 60, seed: int = 42) -> None:
        self.params = dict(params or {})
        self.seq_len = seq_len
        self.seed = seed

    @abstractmethod
    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> Forecaster:
        """Fit on `train` rows; `val` rows are only for early stopping."""

    @abstractmethod
    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        """Predicted next-day log return for each of `rows`, using data up to that row only."""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> Forecaster:
        return joblib.load(directory / "model.joblib")
