from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import CLOSE_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame
from src.features.indicators import add_indicators


def test_columns_and_no_missing_values(ohlcv: pd.DataFrame) -> None:
    frame = build_feature_frame(ohlcv)
    assert list(frame.columns) == [*FEATURE_COLUMNS, CLOSE_COLUMN, TARGET_COLUMN]
    assert np.isfinite(frame.to_numpy()).all()
    assert len(frame) > 0.8 * len(ohlcv)


def test_known_values(ohlcv: pd.DataFrame) -> None:
    frame = build_feature_frame(ohlcv)
    close = ohlcv["Close"]
    day = frame.index[100]
    prev = close.index[close.index.get_loc(day) - 1]
    nxt = close.index[close.index.get_loc(day) + 1]
    assert np.isclose(frame.loc[day, "ret_1"], np.log(close[day] / close[prev]))
    assert np.isclose(frame.loc[day, TARGET_COLUMN], np.log(close[nxt] / close[day]))
    assert np.isclose(frame.loc[day, "mom_5"], np.log(close[day]) - np.log(close.shift(5)[day]))


def test_features_are_causal(ohlcv: pd.DataFrame) -> None:
    """Appending future bars must not change any feature on earlier days."""
    cut = 600
    past = build_feature_frame(ohlcv.iloc[:cut], include_target=False)[FEATURE_COLUMNS]
    full = build_feature_frame(ohlcv, include_target=False)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(past, full.loc[past.index], rtol=1e-9, atol=1e-12)


def test_indicators_are_causal(ohlcv: pd.DataFrame) -> None:
    cut = 500
    past = add_indicators(ohlcv.iloc[:cut])
    full = add_indicators(ohlcv).iloc[:cut]
    pd.testing.assert_frame_equal(past, full, rtol=1e-9, atol=1e-12)


def test_target_is_next_day_and_last_row_handling(ohlcv: pd.DataFrame) -> None:
    with_target = build_feature_frame(ohlcv)
    live = build_feature_frame(ohlcv, include_target=False)
    assert live.index[-1] == ohlcv.index[-1]
    assert np.isnan(live[TARGET_COLUMN].iloc[-1])
    assert with_target.index[-1] == ohlcv.index[-2]
