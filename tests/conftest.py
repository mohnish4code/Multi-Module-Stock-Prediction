"""Offline fixtures: synthetic OHLCV instead of network downloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features.build import build_feature_frame
from src.models.dataset import Dataset


def make_ohlcv(n: int = 900, seed: int = 0, drift: float = 0.0003, vol: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


@pytest.fixture(autouse=True)
def _no_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE", "1")


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def dataset(ohlcv: pd.DataFrame) -> Dataset:
    return Dataset.from_frame("TEST.NS", build_feature_frame(ohlcv))


@pytest.fixture
def cfg() -> dict[str, Any]:
    c = load_config()
    c["backtest"].update({"n_folds": 3, "test_size": 100})
    return c
