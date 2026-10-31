from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.features.build import CLOSE_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class Dataset:
    ticker: str
    dates: pd.DatetimeIndex
    X: np.ndarray       # (n_rows, n_features), unscaled
    y: np.ndarray       # next-day log return per row
    close: np.ndarray   # close on the row's date
    feature_names: list[str]

    @classmethod
    def from_frame(cls, ticker: str, frame: pd.DataFrame, feature_names: list[str] | None = None) -> Dataset:
        names = feature_names or FEATURE_COLUMNS
        return cls(
            ticker=ticker,
            dates=pd.DatetimeIndex(frame.index),
            X=frame[names].to_numpy(dtype="float64"),
            y=frame[TARGET_COLUMN].to_numpy(dtype="float64"),
            close=frame[CLOSE_COLUMN].to_numpy(dtype="float64"),
            feature_names=list(names),
        )

    def col(self, name: str) -> np.ndarray:
        return self.X[:, self.feature_names.index(name)]


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """All length-`seq_len` windows: result[j] covers rows j .. j+seq_len-1, shape (n, seq_len, F)."""
    return sliding_window_view(X, seq_len, axis=0).transpose(0, 2, 1)


def windows_ending_at(windows: np.ndarray, rows: np.ndarray, seq_len: int) -> np.ndarray:
    """Windows whose last row is each of `rows` (needs rows >= seq_len - 1)."""
    return np.ascontiguousarray(windows[rows - seq_len + 1], dtype="float32")
