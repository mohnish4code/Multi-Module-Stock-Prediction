"""OHLCV download with a local parquet snapshot so backtests are reproducible."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_ohlcv(ticker: str, period: str = "15y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return clean_ohlcv(df)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    # Yahoo sometimes returns the latest session with volume but no prices.
    incomplete = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    if incomplete.any():
        logger.warning("Dropping %d bar(s) without prices: %s", incomplete.sum(),
                       ", ".join(str(d.date()) for d in df.index[incomplete][-3:]))
        df = df[~incomplete]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def load_latest(ticker: str, live_dir: Path, period: str = "5y", ttl_minutes: float = 60) -> pd.DataFrame:
    """Recent OHLCV for live forecasts, re-downloaded when the cached copy is older than `ttl_minutes`."""
    path = live_dir / f"{ticker.replace('.', '_')}_{period}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < ttl_minutes * 60:
        return pd.read_parquet(path)
    df = download_ohlcv(ticker, period, "1d")
    live_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_quotes(tickers: list[str]) -> dict[str, dict[str, float]]:
    """Last close and day change for each ticker in one request. Missing tickers are omitted."""
    try:
        data = yf.download(list(tickers), period="7d", interval="1d", auto_adjust=True,
                           progress=False, threads=False, group_by="ticker")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quote fetch failed: %s", exc)
        return {}
    out = {}
    for t in tickers:
        try:
            closes = (data["Close"] if len(tickers) == 1 else data[t]["Close"]).dropna()
        except KeyError:
            continue
        if len(closes) < 2:
            continue
        price, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        out[t] = {"price": price, "prev": prev, "change": price - prev,
                  "change_pct": (price - prev) / prev * 100 if prev else 0.0}
    return out


def load_ohlcv(
    ticker: str,
    raw_dir: Path,
    period: str = "15y",
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """Returns the cached snapshot for `ticker`, downloading it first if absent or `refresh`."""
    path = raw_dir / f"{ticker.replace('.', '_')}_{period}_{interval}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    logger.info("Downloading %s (%s, %s)", ticker, period, interval)
    df = download_ohlcv(ticker, period, interval)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Saved %d rows %s..%s -> %s", len(df), df.index[0].date(), df.index[-1].date(), path)
    return df
