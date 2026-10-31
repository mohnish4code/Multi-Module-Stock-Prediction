"""Technical indicators. Every indicator uses trailing windows only (causal)."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA/EMA/RSI/MACD/Bollinger/volatility columns. Leading NaN rows are kept."""
    df = df.copy()
    close = df["Close"].astype(float)

    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["EMA_20"] = EMAIndicator(close, window=20).ema_indicator()
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()

    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    return df
