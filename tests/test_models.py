from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.baselines import Arima, Majority, Persistence, RandomWalk
from src.models.dataset import Dataset, make_windows, windows_ending_at
from src.models.registry import get_model_class
from src.models.xgb import XGBoost

TRAIN, VAL, TEST = np.arange(59, 500), np.arange(506, 580), np.arange(586, 700)


@pytest.mark.parametrize("cls", [RandomWalk, Persistence, Majority, Arima, XGBoost])
def test_fit_predict_shape_and_finite(cls, dataset: Dataset) -> None:
    model = cls(params={"n_estimators": 50} if cls is XGBoost else None).fit(dataset, TRAIN, VAL)
    pred = model.predict(dataset, TEST)
    assert pred.shape == (len(TEST),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("cls", [Majority, Arima, XGBoost])
def test_save_load_round_trip(cls, dataset: Dataset, tmp_path: Path) -> None:
    model = cls(params={"n_estimators": 50} if cls is XGBoost else None).fit(dataset, TRAIN, VAL)
    model.save(tmp_path)
    loaded = cls.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(dataset, TEST), model.predict(dataset, TEST))


def test_persistence_repeats_todays_return(dataset: Dataset) -> None:
    np.testing.assert_array_equal(Persistence().predict(dataset, TEST), dataset.col("ret_1")[TEST])


def test_arima_prediction_ignores_the_future(dataset: Dataset) -> None:
    """Changing returns after the forecast rows must not change the forecasts."""
    model = Arima().fit(dataset, TRAIN, VAL)
    before = model.predict(dataset, TEST)
    tampered = Dataset(dataset.ticker, dataset.dates, dataset.X, dataset.y.copy(), dataset.close,
                       dataset.feature_names)
    tampered.y[TEST[-1]:] = 0.5
    np.testing.assert_allclose(model.predict(tampered, TEST), before)


def test_arima_recovers_ar1_signal() -> None:
    rng = np.random.default_rng(3)
    n, phi = 1500, 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, 0.01)
    ds = Dataset("AR", None, np.zeros((n, 1)), y, np.ones(n), ["ret_1"])
    model = Arima(params={"max_p": 1, "max_q": 0}).fit(ds, np.arange(0, 1000), np.arange(1000, 1100))
    rows = np.arange(1100, 1500)
    pred = model.predict(ds, rows)
    assert np.corrcoef(pred, y[rows])[0, 1] > 0.35
    np.testing.assert_allclose(pred, phi * y[rows - 1], atol=0.004)


def test_windows_end_at_requested_row() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    w = windows_ending_at(make_windows(X, 3), np.array([2, 9]), 3)
    assert w.shape == (2, 3, 2)
    np.testing.assert_array_equal(w[1], X[7:10])


def test_registry() -> None:
    assert get_model_class("xgboost") is XGBoost
    with pytest.raises(KeyError):
        get_model_class("nope")
