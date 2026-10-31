"""Daily news features built from FinBERT-scored headlines.

Look-ahead rule: a headline published on IST calendar date D only affects
trading days strictly after D. Many historical RSS timestamps carry only a date,
so a same-day headline cannot be placed reliably before or after the 15:30 close;
moving it to the next session guarantees the feature is known at the close of
the day it is attached to.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals.sentiment import finbert_scores
from src.utils import file_lock

logger = logging.getLogger(__name__)

NEWS_COLUMNS = ["news_score", "news_count", "news_score_5d", "has_news"]
IST = "Asia/Kolkata"


def score_with_cache(headlines: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Adds a `score` column; scores are cached by title so reruns only score new headlines."""
    with file_lock(cache_path):
        cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["title", "score"])
    known = dict(zip(cache["title"], cache["score"], strict=True))
    new_titles = [t for t in headlines["title"].unique() if t not in known]
    if new_titles:
        logger.info("FinBERT scoring %d new headlines", len(new_titles))
        known.update(zip(new_titles, finbert_scores(new_titles), strict=True))
        with file_lock(cache_path):  # parallel jobs share the cache: merge with what is on disk now
            if cache_path.exists():
                on_disk = pd.read_parquet(cache_path)
                known = {**dict(zip(on_disk["title"], on_disk["score"], strict=True)), **known}
            pd.DataFrame({"title": list(known), "score": list(known.values())}).to_parquet(cache_path, index=False)
    return headlines.assign(score=headlines["title"].map(known).astype(float))


def daily_features(scored: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per trading day: mean score and log count of the headlines it is allowed to see, a 5-session
    count-weighted mean score, and a has_news flag. Days without news get zeros."""
    out = pd.DataFrame(0.0, index=trading_days, columns=NEWS_COLUMNS)
    if scored.empty:
        return out
    published_day = scored["published"].dt.tz_convert(IST).dt.tz_localize(None).dt.normalize()
    # first trading day strictly after the publication date
    pos = np.searchsorted(trading_days.values, published_day.values, side="right")
    keep = pos < len(trading_days)
    day = pd.Series(trading_days[pos[keep]], index=scored.index[keep])
    grouped = scored.loc[keep, "score"].groupby(day.values)
    total = grouped.sum().reindex(trading_days, fill_value=0.0)
    count = grouped.size().reindex(trading_days, fill_value=0)

    out["news_score"] = np.where(count > 0, total / count.replace(0, 1), 0.0)
    out["news_count"] = np.log1p(count.astype(float))
    roll_total, roll_count = total.rolling(5, min_periods=1).sum(), count.rolling(5, min_periods=1).sum()
    out["news_score_5d"] = np.where(roll_count > 0, roll_total / roll_count.replace(0, 1), 0.0)
    out["has_news"] = (count > 0).astype(float)
    return out
