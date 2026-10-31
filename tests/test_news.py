from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.news_features import NEWS_COLUMNS, daily_features
from src.signals import sentiment

DAYS = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])  # Fri, Mon, Tue, Wed


def headlines(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "title": [f"h{i}" for i in range(len(rows))],
        "published": pd.to_datetime([r[0] for r in rows], utc=True),
        "score": [r[1] for r in rows],
    })


def test_headline_only_affects_later_trading_days() -> None:
    # 2026-01-05 04:00 UTC = 09:30 IST on Monday -> first visible on Tuesday
    out = daily_features(headlines(("2026-01-05 04:00", 0.8)), DAYS)
    assert out.loc["2026-01-05", "has_news"] == 0
    assert out.loc["2026-01-06", "news_score"] == pytest.approx(0.8)
    assert out.loc["2026-01-06", "news_count"] == pytest.approx(np.log1p(1))


def test_ist_date_boundary_and_weekend() -> None:
    # 2026-01-02 20:00 UTC = 01:30 IST on Saturday 3 Jan -> Monday; Sunday news -> Monday too
    out = daily_features(headlines(("2026-01-02 20:00", 0.5), ("2026-01-04 06:00", -0.1)), DAYS)
    assert out.loc["2026-01-02", "has_news"] == 0
    assert out.loc["2026-01-05", "news_score"] == pytest.approx(0.2)
    assert out.loc["2026-01-05", "news_count"] == pytest.approx(np.log1p(2))


def test_rolling_score_and_empty_days() -> None:
    out = daily_features(headlines(("2026-01-01 06:00", 1.0), ("2026-01-05 06:00", -0.5)), DAYS)
    assert list(out.columns) == NEWS_COLUMNS
    assert out.loc["2026-01-02", "news_score_5d"] == pytest.approx(1.0)
    assert out.loc["2026-01-06", "news_score_5d"] == pytest.approx(0.25)  # (1.0 - 0.5) / 2
    assert out.loc["2026-01-07", "news_score"] == 0
    assert daily_features(headlines(), DAYS).to_numpy().sum() == 0


def test_news_after_last_day_is_dropped() -> None:
    out = daily_features(headlines(("2026-01-07 06:00", 0.9)), DAYS)
    assert out["has_news"].sum() == 0


def test_summarise_with_finbert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentiment, "_finbert_or_none", lambda titles: [0.9, -0.1, 0.4][: len(titles)])
    items = [{"title": t, "published": None, "age_days": None} for t in ("a", "b", "c")]
    s = sentiment.summarise(items)
    assert s["method"] == "finbert"
    assert [i["sentiment"] for i in s["news"]] == ["POSITIVE", "NEUTRAL", "POSITIVE"]
    assert s["sentiment_score"] == pytest.approx(0.4)
    assert s["sentiment_label"] == "POSITIVE"


def test_summarise_falls_back_to_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sentiment, "_finbert_or_none", lambda titles: None)
    s = sentiment.summarise([{"title": "Profit surges", "published": None, "age_days": None}])
    assert s["method"] == "keyword"
    assert s["sentiment_label"] == "POSITIVE"
