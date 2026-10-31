"""Naive baselines and ARIMA."""

from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np

from src.models.base import Forecaster
from src.models.dataset import Dataset

logger = logging.getLogger(__name__)


class RandomWalk(Forecaster):
    """Tomorrow's price = today's price (predicted return 0). Price baseline; makes no direction call."""

    name = "random_walk"

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> RandomWalk:
        return self

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return np.zeros(len(rows))


class Persistence(Forecaster):
    """Tomorrow's return = today's return (repeats yesterday's direction)."""

    name = "persistence"

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> Persistence:
        return self

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return ds.col("ret_1")[rows].copy()


class Majority(Forecaster):
    """Always predicts the training set's most frequent direction, with the training mean magnitude."""

    name = "majority"

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> Majority:
        y = ds.y[train]
        sign = 1.0 if np.mean(y > 0) >= 0.5 else -1.0
        self.value_ = sign * max(abs(float(np.mean(y))), 1e-6)
        return self

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.value_)


class Arima(Forecaster):
    """ARMA(p, q) with constant on the next-day return series, order chosen by AIC.

    Prediction runs the fitted parameters through the series from the first
    training row (no refit). y[t-1] equals today's return, which is known at the
    close of day t, so the forecast of y[t] uses no future data.
    """

    name = "arima"

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> Arima:
        from statsmodels.tsa.arima.model import ARIMA

        y = ds.y[train]
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orders = itertools.product(range(self.params.get("max_p", 2) + 1), range(self.params.get("max_q", 2) + 1))
            for p, q in orders:
                try:
                    res = ARIMA(y, order=(p, 0, q), trend="c").fit()
                except Exception as exc:  # noqa: BLE001 - some orders fail to converge
                    logger.debug("ARIMA%s skipped: %s", (p, 0, q), exc)
                    continue
                if best is None or res.aic < best.aic:
                    best = res
        if best is None:
            raise RuntimeError("No ARIMA order converged")
        self.order_ = best.model.order
        self.coef_ = np.asarray(best.params)
        self.start_ = int(train[0])
        logger.debug("ARIMA order %s", self.order_)
        return self

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        last = int(np.max(rows))
        # Run the filter from the training start; on a shorter (live) dataset start where it can.
        start = min(self.start_, int(np.min(rows)))
        history = ds.y[start:last]  # every value here is known at the close of `last`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(history, order=self.order_, trend="c").filter(self.coef_)
            one_step = np.concatenate([np.asarray(res.predict()), np.asarray(res.forecast(1))])
        return one_step[rows - start]
