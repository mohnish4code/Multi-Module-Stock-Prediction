from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    direction_metrics,
    forecast_metrics,
    max_drawdown,
    strategy_metrics,
    strategy_returns,
)


def test_forecast_metrics_random_walk_by_hand() -> None:
    close = np.array([100.0, 200.0])
    y = np.log(np.array([110.0, 180.0]) / close)
    m = forecast_metrics(y, np.zeros(2), close)
    assert m["mae"] == pytest.approx(15.0)
    assert m["rmse"] == pytest.approx(math.sqrt((100 + 400) / 2))
    assert m["mape"] == pytest.approx((10 / 110 + 20 / 180) / 2 * 100)


def test_direction_accuracy_and_pvalue() -> None:
    y = np.array([0.01, -0.01, 0.02, -0.02])
    m = direction_metrics(y, np.array([0.1, 0.1, 0.1, -0.1]))
    assert m["dir_acc"] == 75.0
    assert m["p_vs_50"] == pytest.approx(5 / 16)


def test_all_zero_forecast_makes_no_direction_call() -> None:
    assert math.isnan(direction_metrics(np.array([0.01, -0.01]), np.zeros(2))["dir_acc"])


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([1.1, 0.55, 1.2])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.8])) == pytest.approx(-0.2)


def test_strategy_positions_and_costs() -> None:
    y = np.log(np.array([1.10, 0.90, 1.05]))
    pred = np.array([1.0, -1.0, 1.0])
    daily, pos = strategy_returns(y, pred, threshold=0.0, cost_per_side=0.01)
    assert pos.tolist() == [1, 0, 1]
    # day 1: long, enter (cost); day 2: flat, exit (cost); day 3: re-enter
    assert daily == pytest.approx([0.10 - 0.01, -0.01, 0.05 - 0.01])


def test_strategy_metrics_buy_and_hold_and_sharpe() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0.0005, 0.01, 500)
    m = strategy_metrics(y, np.ones_like(y), cost_per_side=0.0)
    assert m["strat_cagr"] == pytest.approx(m["bh_cagr"])
    assert m["exposure"] == 1.0
    assert m["n_trades"] == 1
    simple = np.expm1(y)
    assert m["strat_sharpe"] == pytest.approx(simple.mean() / simple.std(ddof=1) * math.sqrt(252))
