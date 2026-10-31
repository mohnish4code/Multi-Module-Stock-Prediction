from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor

from src.models.base import Forecaster
from src.models.dataset import Dataset

N_LAGS = 4


def tabular_features(ds: Dataset) -> np.ndarray:
    """Today's features plus the previous N_LAGS daily returns (rows < N_LAGS get NaN lags)."""
    ret = ds.col("ret_1")
    lags = np.full((len(ret), N_LAGS), np.nan)
    for k in range(1, N_LAGS + 1):
        lags[k:, k - 1] = ret[:-k]
    return np.hstack([ds.X, lags])


class XGBoost(Forecaster):
    name = "xgboost"

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> XGBoost:
        params = dict(self.params)
        early = params.pop("early_stopping_rounds", 50)
        X = tabular_features(ds)
        self.model_ = XGBRegressor(
            **params,
            early_stopping_rounds=early,
            random_state=self.seed,
            n_jobs=-1,
            tree_method="hist",
        )
        self.model_.fit(X[train], ds.y[train], eval_set=[(X[val], ds.y[val])], verbose=False)
        return self

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return self.model_.predict(tabular_features(ds)[rows]).astype("float64")
