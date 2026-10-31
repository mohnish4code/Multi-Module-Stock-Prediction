"""Walk-forward backtest: every model is refit on each fold and scored out-of-sample."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.data.loader import load_ohlcv
from src.evaluation import metrics as M
from src.evaluation.splits import Fold, walk_forward_folds
from src.features.build import FEATURE_COLUMNS, build_feature_frame
from src.models.dataset import Dataset
from src.models.registry import create_model
from src.utils import file_lock

logger = logging.getLogger(__name__)


def news_daily_features(cfg: dict[str, Any], ticker: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """FinBERT daily news features from the backfilled headlines (see src.data.news).

    Without a local backfill, the committed results/news_features/<TICKER>.parquet is used, so a
    fresh clone can train the news model without re-downloading years of headlines.
    """
    from src.data.news import RAW_DIR, load_headlines
    from src.features.news_features import NEWS_COLUMNS, daily_features, score_with_cache

    folder = cfg["tickers"][ticker]["folder"]
    saved = resolve_path(cfg["output"]["results_dir"]) / "news_features" / f"{folder}.parquet"
    headlines = load_headlines(folder, resolve_path(RAW_DIR))
    if headlines.empty:
        if not saved.exists():
            raise FileNotFoundError(f"No news data for {folder}. Run: python -m src.data.news")
        logger.info("%s: no local headlines, using saved daily news features", folder)
        return pd.read_parquet(saved)[NEWS_COLUMNS].reindex(trading_days, fill_value=0.0)

    daily = daily_features(score_with_cache(headlines, resolve_path(cfg["news"]["score_cache"])), trading_days)
    saved.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(saved)
    return daily


def load_dataset(cfg: dict[str, Any], ticker: str, refresh: bool = False, feature_set: str = "base") -> Dataset:
    """feature_set 'base' = price/indicator features; 'news' adds the FinBERT daily news features."""
    data = cfg["data"]
    ohlcv = load_ohlcv(ticker, resolve_path(data["raw_dir"]), data["period"], data["interval"], refresh)
    frame = build_feature_frame(ohlcv)
    if feature_set == "base":
        return Dataset.from_frame(ticker, frame)
    if feature_set == "news":
        from src.features.news_features import NEWS_COLUMNS

        frame = frame.join(news_daily_features(cfg, ticker, pd.DatetimeIndex(frame.index)))
        return Dataset.from_frame(ticker, frame, [*FEATURE_COLUMNS, *NEWS_COLUMNS])
    raise ValueError(f"Unknown feature set '{feature_set}'")


def make_folds(cfg: dict[str, Any], n_rows: int) -> list[Fold]:
    bt = cfg["backtest"]
    return walk_forward_folds(
        n_rows=n_rows,
        n_folds=bt["n_folds"],
        test_size=bt["test_size"],
        val_fraction=bt["val_fraction"],
        embargo=bt["embargo"],
        min_start=cfg["features"]["sequence_length"] - 1,
    )


def predict_walk_forward(cfg: dict[str, Any], ds: Dataset, models: list[str], suffix: str = "") -> pd.DataFrame:
    """Long frame of out-of-sample predictions: date, fold, model, pred, actual, close.

    `suffix` is appended to the model label (e.g. '+news' for the news feature set).
    """
    folds = make_folds(cfg, len(ds.y))
    seed = cfg["backtest"]["seed"]
    parts = []
    for name in models:
        for fold in folds:
            started = time.perf_counter()
            model = create_model(cfg, name, seed=seed + fold.fold).fit(ds, fold.train, fold.val)
            pred = model.predict(ds, fold.test)
            parts.append(pd.DataFrame({
                "date": ds.dates[fold.test],
                "fold": fold.fold,
                "model": name + suffix,
                "pred": pred,
                "actual": ds.y[fold.test],
                "close": ds.close[fold.test],
            }))
            logger.info(
                "%s %-11s fold %d  train=%d val=%d test=%d  %.1fs",
                ds.ticker, name + suffix, fold.fold, len(fold.train), len(fold.val), len(fold.test),
                time.perf_counter() - started,
            )
    return pd.concat(parts, ignore_index=True)


def _score(g: pd.DataFrame, baseline_acc: float | None, cfg: dict[str, Any]) -> dict[str, Any]:
    y, p, c = g["actual"].to_numpy(), g["pred"].to_numpy(), g["close"].to_numpy()
    st = cfg["strategy"]
    row: dict[str, Any] = {
        "test_start": g["date"].min().date(),
        "test_end": g["date"].max().date(),
        "n": len(g),
    }
    row.update(M.forecast_metrics(y, p, c))
    row.update(M.direction_metrics(y, p, baseline_acc))
    row.update(M.strategy_metrics(
        y, p, st["signal_threshold"], st["cost_per_side"], st["risk_free_rate"], st["periods_per_year"]
    ))
    threshold = cfg["signals"]["neutral_threshold_pct"]
    row["pct_days_above_neutral"] = float(np.mean(np.abs(np.expm1(p)) * 100 > threshold) * 100)
    return row


def score_predictions(preds: pd.DataFrame, ticker: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per (model, fold) plus fold='all' on the concatenated out-of-sample period."""
    preds = preds.sort_values("date")
    majority = preds[preds["model"] == "majority"]

    def majority_acc(fold: int | None) -> float | None:
        g = majority if fold is None else majority[majority["fold"] == fold]
        if g.empty:
            return None
        return float(np.mean((g["pred"] > 0) == (g["actual"] > 0)) * 100)

    rows = []
    for model, gm in preds.groupby("model", sort=False):
        for fold, gf in gm.groupby("fold"):
            rows.append({"ticker": ticker, "model": model, "fold": str(fold),
                         **_score(gf, majority_acc(int(fold)), cfg)})
        rows.append({"ticker": ticker, "model": model, "fold": "all", **_score(gm, majority_acc(None), cfg)})
    return pd.DataFrame(rows)


def merge_into_csv(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Replaces rows for the (ticker, model) pairs in `new`, keeps everything else."""
    with file_lock(path):
        if path.exists():
            old = pd.read_csv(path, dtype={"fold": str})
            keys = set(zip(new["ticker"], new["model"], strict=True))
            keep = ~pd.Series(list(zip(old["ticker"], old["model"], strict=True))).isin(keys).to_numpy()
            new = pd.concat([old[keep], new], ignore_index=True)
        new = new.sort_values(["ticker", "model", "fold"], kind="stable")
        new.to_csv(path, index=False, float_format="%.6g")
    return new


def merge_predictions(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["model"].isin(new["model"].unique())], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new
