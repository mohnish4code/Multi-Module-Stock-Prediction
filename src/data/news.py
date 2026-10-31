"""Company headlines from Google News RSS: the live feed and the weekly historical backfill.

    python -m src.data.news                  # backfill all tickers from the price snapshot start
    python -m src.data.news --tickers TCS

The backfill caches each week as data/news/raw/<FOLDER>/<week_start>.json (an
empty list when nothing was found), so it can be stopped and resumed. Requests
are spaced out and back off on HTTP 429/5xx.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = "data/news/raw"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def _published(entry: Any) -> dt.datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        st = entry.get(attr)
        if st:
            try:
                return dt.datetime(*st[:6], tzinfo=dt.UTC)
            except (TypeError, ValueError):
                pass
    return None


def fetch_feed(url: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - fixed https host
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return feedparser.parse(resp.read())


def search_url(query: str, window: str | None = "1y", after: str | None = None, before: str | None = None) -> str:
    terms = quote(query)
    if after and before:
        terms += f"+after:{after}+before:{before}"
    elif window:
        terms += f"+when:{window}"
    return f"https://news.google.com/rss/search?q={terms}&hl=en-IN&gl=IN&ceid=IN:en"


def parse_entries(feed: Any, now: dt.datetime | None = None) -> list[dict[str, Any]]:
    now = now or dt.datetime.now(dt.UTC)
    items = []
    for entry in getattr(feed, "entries", []):
        title = entry.get("title", "").strip()
        if not title:
            continue
        published = _published(entry)
        source = entry.get("source", {})
        items.append({
            "title": title,
            "publisher": source.get("title", "Unknown") if isinstance(source, dict) else "Unknown",
            "link": entry.get("link", ""),
            "published": published.isoformat() if published else None,
            "age_days": round((now - published).total_seconds() / 86400, 1) if published else None,
        })
    items.sort(key=lambda it: it["published"] or "", reverse=True)
    return items


# ---------------------------------------------------------------- live feed

def get_company_news(
    query: str, max_items: int = 15, recency_days: float = 21, min_items: int = 4
) -> list[dict[str, Any]]:
    """Headlines newer than `recency_days`; if fewer than `min_items`, the newest available instead."""
    try:
        items = parse_entries(fetch_feed(search_url(query)))
    except Exception as exc:  # noqa: BLE001 - news is optional; never break a prediction on it
        logger.warning("News fetch failed for %r: %s", query, exc)
        return []
    fresh = [it for it in items if it["age_days"] is not None and it["age_days"] <= recency_days]
    return (fresh if len(fresh) >= min_items else items)[:max_items]


# ---------------------------------------------------------------- historical backfill

def fetch_week(query: str, start: pd.Timestamp, max_tries: int = 6) -> list[dict[str, Any]]:
    url = search_url(query, after=str(start.date()), before=str((start + pd.Timedelta(days=7)).date()))
    delay = 30.0
    for attempt in range(1, max_tries + 1):
        try:
            items = parse_entries(fetch_feed(url, timeout=20))
            return [{k: it[k] for k in ("title", "publisher", "link", "published")} for it in items]
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == max_tries:
                raise
            logger.warning("HTTP %s for %s; sleeping %.0fs", exc.code, start.date(), delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_tries:
                raise
            logger.warning("%s for %s; sleeping %.0fs", exc, start.date(), delay)
        time.sleep(delay)
        delay = min(delay * 2, 900)
    return []


def backfill(folder: str, query: str, start: pd.Timestamp, end: pd.Timestamp, out_dir: Path,
             pause: float = 1.2) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks = pd.date_range(start.normalize(), end, freq="7D")
    fetched = 0
    for i, week in enumerate(weeks, 1):
        path = out_dir / f"{week.date()}.json"
        if path.exists():
            continue
        path.write_text(json.dumps(fetch_week(query, week), ensure_ascii=False), encoding="utf-8")
        fetched += 1
        if fetched % 50 == 0:
            logger.info("%s: %d/%d weeks", folder, i, len(weeks))
        time.sleep(pause)
    return fetched


def load_headlines(folder: str, raw_dir: Path) -> pd.DataFrame:
    """All cached headlines for a ticker, de-duplicated by title, with a UTC `published` timestamp."""
    rows = [it for f in sorted((raw_dir / folder).glob("*.json")) for it in json.loads(f.read_text(encoding="utf-8"))]
    df = pd.DataFrame(rows, columns=["title", "publisher", "link", "published"])
    df = df.dropna(subset=["published"]).drop_duplicates(subset=["title"])
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.sort_values("published").reset_index(drop=True)


def main() -> None:
    from src.config import load_config, resolve_path, resolve_ticker
    from src.data.loader import load_ohlcv
    from src.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--start", default=None, help="default: first date of the price snapshot")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    tickers = [resolve_ticker(cfg, t) for t in args.tickers] if args.tickers else list(cfg["tickers"])
    end = pd.Timestamp.now().normalize()
    for ticker in tickers:
        info, data = cfg["tickers"][ticker], cfg["data"]
        start = (pd.Timestamp(args.start) if args.start else
                 load_ohlcv(ticker, resolve_path(data["raw_dir"]), data["period"], data["interval"]).index[0])
        n = backfill(info["folder"], info["history_query"], start, end, resolve_path(RAW_DIR) / info["folder"])
        logger.info("%s done (%d new weeks)", info["folder"], n)


if __name__ == "__main__":
    main()
