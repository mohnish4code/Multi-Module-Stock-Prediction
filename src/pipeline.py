"""End-to-end analysis for one ticker. The CLI, the API and (through the API) the
Streamlit app all call these functions, so they always agree.

Every return value is JSON-serialisable.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.loader import load_latest
from src.data.news import fetch_week, get_company_news
from src.features.build import build_feature_frame
from src.features.indicators import add_indicators
from src.models.base import Forecaster
from src.models.dataset import Dataset
from src.models.registry import load_model, variant
from src.signals import sentiment
from src.signals.decision import classify_direction, decide
from src.signals.technical import analyze_technicals

logger = logging.getLogger(__name__)


NEWS_TTL_SECONDS = 3600
_news_cache: dict[str, tuple[float, pd.DataFrame]] = {}


@lru_cache(maxsize=64)
def _cached_model(models_root: str, folder: str, label: str) -> tuple[Forecaster, dict[str, Any]]:
    return load_model(resolve_path(models_root), folder, label)


def serving_label(cfg: dict[str, Any]) -> str:
    serving = cfg["serving"]
    return variant(serving["model"], serving.get("feature_set", "base"))


def serving_model(cfg: dict[str, Any], ticker: str, label: str | None = None) -> tuple[Forecaster, dict[str, Any]]:
    return _cached_model(cfg["serving"]["models_dir"], cfg["tickers"][ticker]["folder"], label or serving_label(cfg))


def recent_headlines(cfg: dict[str, Any], ticker: str, end: pd.Timestamp, weeks: int = 3) -> pd.DataFrame:
    """Headlines from the same weekly dated queries the news model was trained on (cached for an hour).
    A failed fetch returns what was collected so far; days without news are a normal input."""
    info = cfg["tickers"][ticker]
    cached = _news_cache.get(info["folder"])
    if cached and time.monotonic() - cached[0] < NEWS_TTL_SECONDS:
        return cached[1]
    rows: list[dict[str, Any]] = []
    for start in pd.date_range(end.normalize() - pd.Timedelta(weeks=weeks), end, freq="7D"):
        try:
            rows += fetch_week(info["history_query"], start, max_tries=1)
        except Exception as exc:  # noqa: BLE001 - serve without news rather than fail
            logger.warning("News fetch for %s failed: %s", info["folder"], exc)
            break
    df = pd.DataFrame(rows, columns=["title", "publisher", "link", "published"]).dropna(subset=["published"])
    df = df.drop_duplicates(subset=["title"]).assign(published=lambda d: pd.to_datetime(d["published"], utc=True))
    _news_cache[info["folder"]] = (time.monotonic(), df)
    return df


def live_news_features(cfg: dict[str, Any], ticker: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    from src.features.news_features import daily_features, score_with_cache

    headlines = recent_headlines(cfg, ticker, days[-1])
    if headlines.empty:
        return daily_features(headlines.assign(score=[]), days)
    return daily_features(score_with_cache(headlines, resolve_path(cfg["news"]["score_cache"])), days)


def predict(cfg: dict[str, Any], ticker: str, ohlcv: pd.DataFrame, label: str | None = None) -> dict[str, Any]:
    """Next-day forecast from the latest bar of `ohlcv`."""
    model, manifest = serving_model(cfg, ticker, label)
    frame = build_feature_frame(ohlcv, include_target=False)
    if manifest.get("feature_set") == "news":
        frame = frame.join(live_news_features(cfg, ticker, pd.DatetimeIndex(frame.index)))
    ds = Dataset.from_frame(ticker, frame, manifest.get("features"))
    last = len(ds.y) - 1
    pred_return = float(model.predict(ds, np.array([last]))[0])

    current = float(ds.close[last])
    predicted = current * float(np.exp(pred_return))
    change_pct = (predicted / current - 1) * 100
    return {
        "model": manifest["model"],
        "model_trained_at": manifest["trained_at"],
        "as_of": ds.dates[last].date().isoformat(),
        "current_price": current,
        "predicted_price": predicted,
        "predicted_return": pred_return,
        "predicted_change": predicted - current,
        "predicted_change_pct": change_pct,
        "direction": classify_direction(change_pct, cfg["signals"]["neutral_threshold_pct"]),
    }


def news_sentiment(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    nc = cfg["news"]
    items = get_company_news(cfg["tickers"][ticker]["news_query"], nc["max_items"], nc["recency_days"], nc["min_items"])
    summary = sentiment.summarise(items, method=nc.get("method", "finbert"))
    return {**summary, "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


def price_history(indicators: pd.DataFrame, bars: int) -> list[dict[str, Any]]:
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            "SMA_20": "sma_20", "EMA_20": "ema_20", "BB_Upper": "bb_upper", "BB_Lower": "bb_lower"}
    tail = indicators[list(cols)].tail(bars).rename(columns=cols)
    tail = tail.astype(float).round(4).replace({np.nan: None})
    return [{"date": d.date().isoformat(), **row} for d, row in zip(tail.index, tail.to_dict("records"), strict=True)]


def analyse(
    ticker: str, cfg: dict[str, Any] | None = None, include_news: bool = True, ohlcv: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Forecast + technical read + news sentiment + the single decision engine."""
    cfg = cfg or load_config()
    data = cfg["data"]
    if ohlcv is None:
        ohlcv = load_latest(ticker, resolve_path(data["live_dir"]), data["live_period"], data["live_ttl_minutes"])

    forecast = predict(cfg, ticker, ohlcv)
    indicators = add_indicators(ohlcv)
    technical = analyze_technicals(indicators)
    news = news_sentiment(cfg, ticker) if include_news else sentiment.summarise([], method="keyword")

    decision = decide(
        model_direction=forecast["direction"],
        predicted_change_pct=forecast["predicted_change_pct"],
        technical_signal=technical["technical_signal"],
        news_sentiment=news["sentiment_label"],
        volatility_level=technical["volatility_level"],
    )
    return {
        "ticker": ticker,
        "name": cfg["tickers"][ticker]["name"],
        "forecast": forecast,
        "technical": technical,
        "news": news,
        "decision": decision,
        "history": price_history(indicators, cfg["serving"]["history_bars"]),
    }


def backtest_summary(ticker: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Walk-forward results for every model on this ticker (fold='all' and per fold)."""
    cfg = cfg or load_config()
    folder = cfg["tickers"][ticker]["folder"]
    results = resolve_path(cfg["output"]["results_dir"])
    metrics = pd.read_csv(results / "metrics.csv", dtype={"fold": str})
    metrics = metrics[metrics["ticker"] == folder]
    if metrics.empty:
        raise FileNotFoundError(f"No backtest results for {folder}. Run: python backtest.py --tickers {folder}")

    preds = pd.read_parquet(results / "predictions" / f"{folder}.parquet").sort_values("date")
    st = cfg["strategy"]
    from src.evaluation.metrics import strategy_returns

    curves: dict[str, list[float]] = {}
    dates: list[str] = []
    for model, g in preds.groupby("model", sort=False):
        daily, _ = strategy_returns(g["actual"].to_numpy(), g["pred"].to_numpy(),
                                    st["signal_threshold"], st["cost_per_side"])
        curves[model] = np.round(np.cumprod(1 + daily), 5).tolist()
        if not dates:
            y = g["actual"].to_numpy()
            bh, _ = strategy_returns(y, np.ones_like(y), 0.0, st["cost_per_side"])
            curves["buy_and_hold"] = np.round(np.cumprod(1 + bh), 5).tolist()
            dates = [d.date().isoformat() for d in pd.to_datetime(g["date"])]

    records = metrics.replace({np.nan: None}).to_dict("records")
    return {
        "ticker": ticker,
        "cost_per_side": st["cost_per_side"],
        "metrics": records,
        "equity": {"dates": dates, "curves": curves},
    }
