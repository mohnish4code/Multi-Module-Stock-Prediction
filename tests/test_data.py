from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import clean_ohlcv
from src.data.news import parse_entries, search_url


def test_clean_ohlcv_flattens_and_drops_priceless_bars(ohlcv: pd.DataFrame) -> None:
    raw = ohlcv.copy()
    raw.loc[raw.index[-1], ["Open", "High", "Low", "Close"]] = np.nan  # volume-only bar
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["TCS.NS"]])
    out = clean_ohlcv(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index[-1] == ohlcv.index[-2]
    assert out.index.is_monotonic_increasing


def test_clean_ohlcv_requires_columns(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        clean_ohlcv(ohlcv.drop(columns="Volume"))


def test_parse_entries_sorts_newest_first() -> None:
    class Feed:
        entries = [
            {"title": "Old", "published_parsed": (2026, 1, 1, 0, 0, 0), "source": {"title": "A"}},
            {"title": " ", "published_parsed": (2026, 1, 3, 0, 0, 0)},
            {"title": "New", "published_parsed": (2026, 1, 2, 0, 0, 0)},
        ]

    items = parse_entries(Feed(), now=pd.Timestamp("2026-01-03", tz="UTC").to_pydatetime())
    assert [it["title"] for it in items] == ["New", "Old"]
    assert items[0]["age_days"] == 1.0
    assert items[1]["publisher"] == "A"


def test_search_url_window_and_date_range() -> None:
    assert "when:1y" in search_url("TCS share")
    url = search_url('"HDFC Bank"', after="2020-01-01", before="2020-01-08")
    assert "after:2020-01-01+before:2020-01-08" in url
    assert "when:" not in url
