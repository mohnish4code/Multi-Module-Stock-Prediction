"""Stationary model features built from the indicator frame.

Absolute prices drift upward, so the models see returns and ratios instead;
these stay in a stable range for the life of a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import add_indicators

FEATURE_COLUMNS: list[str] = [
    "ret_1",        # log return of Close vs previous Close
    "ret_open",     # overnight gap: log(Open / prev Close)
    "hl_range",     # (High - Low) / prev Close
    "co_range",     # (Close - Open) / Open
    "vol_ratio",    # log1p(Volume / 20d average Volume)
    "rsi",          # RSI / 100
    "macd_n",       # MACD / Close
    "macd_sig_n",   # MACD signal / Close
    "macd_hist_n",  # MACD histogram / Close
    "sma20_gap",    # (Close - SMA_20) / SMA_20
    "ema20_gap",    # (Close - EMA_20) / EMA_20
    "sma_cross",    # (SMA_20 - SMA_50) / SMA_50
    "bb_pctb",      # position of Close within the Bollinger band
    "volatility",   # 20d rolling std of daily returns
    "mom_5",        # 5-day cumulative log return
    "mom_10",
    "mom_20",
    "dist_hi_252",  # Close / 252-day high - 1
    "dist_lo_252",  # Close / 252-day low - 1
    "dow_sin",      # day-of-week, cyclical
    "dow_cos",
]

TARGET_COLUMN = "target"  # next day's log return of Close
CLOSE_COLUMN = "close"


def build_feature_frame(ohlcv: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """OHLCV -> FEATURE_COLUMNS + close + target. Rows with missing features are dropped.

    include_target=True also drops the last row (no next day yet). With False it
    is kept with a NaN target, which is the row a live forecast is made from.
    """
    df = add_indicators(ohlcv)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    prev_close = close.shift(1)
    volume = df["Volume"].astype(float)
    bb_width = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    log_close = np.log(close)

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = np.log(close / prev_close)
    out["ret_open"] = np.log(open_ / prev_close)
    out["hl_range"] = (df["High"] - df["Low"]) / prev_close
    out["co_range"] = (close - open_) / open_
    out["vol_ratio"] = np.log1p(volume / volume.rolling(20).mean())
    out["rsi"] = df["RSI"] / 100.0
    out["macd_n"] = df["MACD"] / close
    out["macd_sig_n"] = df["MACD_Signal"] / close
    out["macd_hist_n"] = df["MACD_Histogram"] / close
    out["sma20_gap"] = (close - df["SMA_20"]) / df["SMA_20"]
    out["ema20_gap"] = (close - df["EMA_20"]) / df["EMA_20"]
    out["sma_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
    out["bb_pctb"] = (close - df["BB_Lower"]) / bb_width
    out["volatility"] = df["Volatility"]
    for n in (5, 10, 20):
        out[f"mom_{n}"] = log_close - log_close.shift(n)
    out["dist_hi_252"] = close / close.rolling(252, min_periods=60).max() - 1.0
    out["dist_lo_252"] = close / close.rolling(252, min_periods=60).min() - 1.0
    dow = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    out[CLOSE_COLUMN] = close
    out[TARGET_COLUMN] = np.log(close.shift(-1) / close)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURE_COLUMNS, CLOSE_COLUMN])
    return out.dropna(subset=[TARGET_COLUMN]) if include_target else out
