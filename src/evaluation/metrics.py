"""Forecast, direction and long/flat strategy metrics.

All inputs are next-day log returns aligned by row: y_true[i] is the realised
return after day i, y_pred[i] the forecast made at the close of day i.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, close: np.ndarray) -> dict[str, float]:
    """Errors on the reconstructed next-day price."""
    actual = close * np.exp(y_true)
    predicted = close * np.exp(y_pred)
    err = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / actual) * 100.0),
    }


def direction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, baseline_acc: float | None = None
) -> dict[str, float]:
    """Up / not-up accuracy with one-sided binomial p-values.

    A model whose forecasts are all exactly zero makes no directional call,
    so its direction metrics are NaN.
    """
    if np.all(y_pred == 0):
        return {"dir_acc": math.nan, "p_vs_50": math.nan, "p_vs_baseline": math.nan}

    hits = int(np.sum((y_pred > 0) == (y_true > 0)))
    n = len(y_true)
    out = {
        "dir_acc": hits / n * 100.0,
        "p_vs_50": binomtest(hits, n, 0.5, alternative="greater").pvalue,
        "p_vs_baseline": math.nan,
    }
    if baseline_acc is not None and 0 < baseline_acc < 100:
        out["p_vs_baseline"] = binomtest(hits, n, baseline_acc / 100.0, alternative="greater").pvalue
    return out


def max_drawdown(equity: np.ndarray) -> float:
    curve = np.concatenate([[1.0], equity])
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def _perf(daily: np.ndarray, periods_per_year: int, risk_free_rate: float) -> dict[str, float]:
    equity = np.cumprod(1.0 + daily)
    n = len(daily)
    std = float(np.std(daily, ddof=1)) if n > 1 else 0.0
    excess = daily - risk_free_rate / periods_per_year
    return {
        "total_return": float(equity[-1] - 1.0),
        "cagr": float(equity[-1] ** (periods_per_year / n) - 1.0),
        "sharpe": float(np.mean(excess) / std * math.sqrt(periods_per_year)) if std > 0 else math.nan,
        "max_dd": max_drawdown(equity),
    }


def strategy_returns(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0, cost_per_side: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Long/flat: hold the stock for day i+1 when y_pred[i] > threshold, else cash.

    Returns (daily net simple returns, positions). Costs are charged on each entry/exit.
    """
    position = (y_pred > threshold).astype(float)
    trades = np.abs(np.diff(position, prepend=0.0))
    daily = position * np.expm1(y_true) - cost_per_side * trades
    return daily, position


def strategy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0,
    cost_per_side: float = 0.0012,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    net, position = strategy_returns(y_true, y_pred, threshold, cost_per_side)
    gross, _ = strategy_returns(y_true, y_pred, threshold, 0.0)
    bh, _ = strategy_returns(y_true, np.ones_like(y_true), 0.0, cost_per_side)

    out: dict[str, float] = {}
    out.update({f"strat_{k}": v for k, v in _perf(net, periods_per_year, risk_free_rate).items()})
    out["strat_sharpe_gross"] = _perf(gross, periods_per_year, risk_free_rate)["sharpe"]
    out["strat_cagr_gross"] = _perf(gross, periods_per_year, risk_free_rate)["cagr"]
    out.update({f"bh_{k}": v for k, v in _perf(bh, periods_per_year, risk_free_rate).items()})
    out["excess_cagr_vs_bh"] = out["strat_cagr"] - out["bh_cagr"]
    out["exposure"] = float(position.mean())
    out["n_trades"] = int(np.abs(np.diff(position, prepend=0.0)).sum())
    return out
