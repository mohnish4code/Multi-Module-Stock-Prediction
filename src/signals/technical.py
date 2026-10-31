"""Rule-based technical read of the latest bar (RSI, SMA trend, MACD, volatility)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _vote(bullish: bool, bearish: bool) -> str:
    return "BULLISH" if bullish else "BEARISH" if bearish else "NEUTRAL"


def volatility_level(daily_std: float) -> str:
    if daily_std < 0.01:
        return "LOW"
    if daily_std < 0.03:
        return "MEDIUM"
    return "HIGH"


def analyze_technicals(indicators: pd.DataFrame) -> dict[str, Any]:
    """`indicators` is the output of src.features.indicators.add_indicators."""
    last = indicators.dropna(subset=["RSI", "SMA_50", "MACD_Signal", "Volatility"]).iloc[-1]
    rsi, sma20, sma50 = float(last["RSI"]), float(last["SMA_20"]), float(last["SMA_50"])
    macd, macd_sig, vol = float(last["MACD"]), float(last["MACD_Signal"]), float(last["Volatility"])

    rsi_signal = _vote(rsi < 30, rsi > 70)
    trend = _vote(sma20 > sma50, sma20 < sma50)
    macd_trend = _vote(macd > macd_sig, macd < macd_sig)
    level = volatility_level(vol)

    votes = [rsi_signal, trend, macd_trend]
    bull, bear = votes.count("BULLISH"), votes.count("BEARISH")
    signal = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "NEUTRAL"

    rsi_text = {"BULLISH": "an oversold market", "BEARISH": "an overbought market",
                "NEUTRAL": "neutral conditions"}[rsi_signal]
    return {
        "rsi": rsi,
        "rsi_signal": rsi_signal,
        "sma_20": sma20,
        "sma_50": sma50,
        "trend": trend,
        "macd": macd,
        "macd_signal": macd_sig,
        "macd_trend": macd_trend,
        "volatility": vol,
        "volatility_level": level,
        "technical_signal": signal,
        "bullish_signals": bull,
        "bearish_signals": bear,
        "neutral_signals": votes.count("NEUTRAL"),
        "signals": [
            f"RSI is {rsi:.2f}, indicating {rsi_text}.",
            f"SMA 20 is {'above' if sma20 > sma50 else 'below' if sma20 < sma50 else 'equal to'} SMA 50 "
            f"({trend.lower()} short-term trend).",
            f"MACD is {'above' if macd > macd_sig else 'below' if macd < macd_sig else 'at'} its signal line "
            f"({macd_trend.lower()} momentum).",
            f"20-day volatility is {level} ({vol:.4f}).",
        ],
    }
