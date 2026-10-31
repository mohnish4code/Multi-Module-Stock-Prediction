from __future__ import annotations

import pandas as pd
import pytest

from src.features.indicators import add_indicators
from src.signals.decision import classify_direction, coordinate, decide
from src.signals.sentiment import score_headline, summarise
from src.signals.technical import analyze_technicals, volatility_level


@pytest.mark.parametrize(
    ("pct", "expected"), [(0.2, "BULLISH"), (-0.2, "BEARISH"), (0.05, "NEUTRAL"), (0.1, "NEUTRAL")]
)
def test_classify_direction(pct: float, expected: str) -> None:
    assert classify_direction(pct, 0.1) == expected


@pytest.mark.parametrize(
    ("model", "tech", "news", "signal", "agreement"),
    [
        ("BULLISH", "BULLISH", "POSITIVE", "BULLISH", "STRONG"),
        ("BEARISH", "BEARISH", "NEUTRAL", "BEARISH", "MODERATE"),
        ("BULLISH", "BEARISH", "NEUTRAL", "NEUTRAL", "MIXED"),
        ("NEUTRAL", "NEUTRAL", "NEGATIVE", "BEARISH", "MIXED"),
    ],
)
def test_coordinate(model: str, tech: str, news: str, signal: str, agreement: str) -> None:
    out = coordinate(model, tech, news)
    assert (out["coordinated_signal"], out["agreement"]) == (signal, agreement)
    assert out["bullish_modules"] + out["bearish_modules"] + out["neutral_modules"] == 3


def test_decide_buy_when_all_agree_and_risk_low() -> None:
    d = decide("BULLISH", 0.3, "BULLISH", "POSITIVE", "LOW")
    assert d["recommendation"] == "BUY"
    assert d["confidence_score"] == 100
    assert d["risk_level"] == "LOW"


def test_decide_holds_on_disagreement() -> None:
    assert decide("BULLISH", 0.3, "BEARISH", "NEGATIVE", "HIGH")["recommendation"] == "HOLD"


def test_decide_sell() -> None:
    assert decide("BEARISH", -0.3, "BEARISH", "NEGATIVE", "LOW")["recommendation"] == "SELL"


def test_unknown_sentiment_is_neutral() -> None:
    assert decide("BULLISH", 0.3, "BULLISH", "UNKNOWN", "LOW")["recommendation"] == "BUY"


def test_keyword_sentiment() -> None:
    assert score_headline("Profit surges to record high")[0] == "POSITIVE"
    assert score_headline("Shares plunge on fraud investigation")[0] == "NEGATIVE"
    assert score_headline("Company announces board meeting")[0] == "NEUTRAL"
    s = summarise([{"title": "Profit rises", "published": None, "age_days": None},
                   {"title": "Board meeting", "published": None, "age_days": None}], method="keyword")
    assert s["total_news"] == 2
    assert s["sentiment_label"] == "POSITIVE"
    assert summarise([], method="keyword")["sentiment_label"] == "NEUTRAL"


def test_technicals(ohlcv: pd.DataFrame) -> None:
    out = analyze_technicals(add_indicators(ohlcv))
    assert out["technical_signal"] in ("BULLISH", "BEARISH", "NEUTRAL")
    assert out["bullish_signals"] + out["bearish_signals"] + out["neutral_signals"] == 3
    assert 0 <= out["rsi"] <= 100
    assert len(out["signals"]) == 4
    assert volatility_level(0.005) == "LOW"
    assert volatility_level(0.02) == "MEDIUM"
    assert volatility_level(0.05) == "HIGH"
