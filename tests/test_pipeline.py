"""Backtest scoring, CSV merging, config and the end-to-end analysis (offline, with a tiny model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import resolve_ticker
from src.evaluation.backtest import make_folds, merge_into_csv, predict_walk_forward, score_predictions
from src.models.dataset import Dataset


def test_resolve_ticker(cfg: dict[str, Any]) -> None:
    assert resolve_ticker(cfg, "tcs") == "TCS.NS"
    assert resolve_ticker(cfg, "HDFCBANK") == "HDFCBANK.NS"
    with pytest.raises(KeyError):
        resolve_ticker(cfg, "AAPL")


def test_walk_forward_and_scoring(cfg: dict[str, Any], dataset: Dataset) -> None:
    preds = predict_walk_forward(cfg, dataset, ["random_walk", "majority", "persistence"])
    folds = make_folds(cfg, len(dataset.y))
    assert len(preds) == 3 * sum(len(f.test) for f in folds)
    assert preds["date"].is_monotonic_increasing or preds.groupby("model")["date"].is_monotonic_increasing.all()

    metrics = score_predictions(preds, "TEST", cfg)
    assert set(metrics["fold"]) == {"0", "1", "2", "all"}
    allf = metrics[metrics["fold"] == "all"].set_index("model")
    assert np.isnan(allf.loc["random_walk", "dir_acc"])
    assert allf.loc["random_walk", "exposure"] == 0.0
    assert allf.loc["persistence", "n"] == len(dataset.y) - folds[0].test[0]


def test_merge_into_csv_replaces_only_rerun_pairs(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    merge_into_csv(pd.DataFrame({"ticker": ["A", "B"], "model": ["x", "x"], "fold": ["all", "all"], "v": [1, 2]}), path)
    out = merge_into_csv(pd.DataFrame({"ticker": ["A"], "model": ["x"], "fold": ["all"], "v": [9]}), path)
    assert out.set_index("ticker")["v"].to_dict() == {"A": 9, "B": 2}


def test_analyse_end_to_end(cfg: dict[str, Any], ohlcv: pd.DataFrame, tmp_path: Path, monkeypatch) -> None:
    from src import pipeline
    from src.models.baselines import Majority

    folder = cfg["tickers"]["TCS.NS"]["folder"]
    model_path = tmp_path / folder / "majority"
    ds = Dataset.from_frame("TCS.NS", pipeline.build_feature_frame(ohlcv))
    Majority().fit(ds, np.arange(59, 600), np.arange(610, 700)).save(model_path)
    (model_path / "manifest.json").write_text(json.dumps({"model": "majority", "trained_at": "2026-01-01T00:00"}))

    cfg["serving"].update({"model": "majority", "feature_set": "base", "models_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline, "resolve_path", lambda p: Path(p))
    pipeline._cached_model.cache_clear()

    out = pipeline.analyse("TCS.NS", cfg, include_news=False, ohlcv=ohlcv)
    json.dumps(out)  # must be serialisable for the API
    assert out["forecast"]["as_of"] == ohlcv.index[-1].date().isoformat()
    assert out["forecast"]["current_price"] == pytest.approx(ohlcv["Close"].iloc[-1])
    assert out["decision"]["recommendation"] in ("BUY", "HOLD", "SELL")
    assert len(out["history"]) == cfg["serving"]["history_bars"]


def test_news_model_gets_live_news_features(cfg: dict[str, Any], ohlcv: pd.DataFrame, tmp_path: Path,
                                            monkeypatch) -> None:
    """A model trained with the news feature set is served with the same four columns, built from live headlines."""
    from src import pipeline
    from src.features import news_features
    from src.features.build import FEATURE_COLUMNS
    from src.models.baselines import Majority

    seen: dict[str, Any] = {}

    def spy_predict(self, ds, rows):
        seen["features"] = ds.feature_names
        seen["last_row"] = dict(zip(ds.feature_names, ds.X[rows[-1]], strict=True))
        return np.zeros(len(rows))

    ds = Dataset.from_frame("TCS.NS", pipeline.build_feature_frame(ohlcv))
    out = tmp_path / "TCS" / "majority+news"
    Majority().fit(ds, np.arange(59, 600), np.arange(610, 700)).save(out)
    monkeypatch.setattr(Majority, "predict", spy_predict)
    features = [*FEATURE_COLUMNS, *news_features.NEWS_COLUMNS]
    (out / "manifest.json").write_text(json.dumps({"model": "majority+news", "feature_set": "news",
                                                   "features": features, "trained_at": "2026-01-01T00:00"}))
    day_before = ohlcv.index[-2]
    headlines = pd.DataFrame({"title": ["Profit jumps"], "publisher": ["X"], "link": [""],
                              "published": [pd.Timestamp(day_before, tz="Asia/Kolkata").tz_convert("UTC")]})
    monkeypatch.setattr(pipeline, "recent_headlines", lambda *_a, **_k: headlines)
    monkeypatch.setattr(news_features, "score_with_cache", lambda h, _p: h.assign(score=0.7))
    monkeypatch.setattr(pipeline, "resolve_path", lambda p: Path(p))
    cfg["serving"].update({"model": "majority", "feature_set": "news", "models_dir": str(tmp_path)})
    pipeline._cached_model.cache_clear()

    forecast = pipeline.predict(cfg, "TCS.NS", ohlcv)
    pipeline._cached_model.cache_clear()
    assert forecast["model"] == "majority+news"
    assert seen["features"] == features
    assert seen["last_row"]["news_score"] == pytest.approx(0.7)  # published the day before -> visible today
    assert seen["last_row"]["has_news"] == 1.0


def test_news_fetch_failure_serves_without_news(cfg: dict[str, Any], monkeypatch) -> None:
    from src import pipeline

    def down(*_a, **_k):
        raise OSError("network down")

    monkeypatch.setattr(pipeline, "fetch_week", down)
    pipeline._news_cache.clear()
    days = pd.bdate_range("2026-01-01", periods=10)
    out = pipeline.live_news_features(cfg, "TCS.NS", days)
    pipeline._news_cache.clear()
    assert out.to_numpy().sum() == 0


def test_news_features_fall_back_to_saved_file(cfg: dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    """A fresh clone has no raw headlines but does have results/news_features/*.parquet."""
    from src.data import news
    from src.evaluation import backtest
    from src.features.news_features import NEWS_COLUMNS

    days = pd.bdate_range("2026-01-01", periods=6)
    saved = pd.DataFrame(0.5, index=days[:4], columns=NEWS_COLUMNS)
    (tmp_path / "news_features").mkdir()
    saved.to_parquet(tmp_path / "news_features" / "TCS.parquet")
    cfg["output"]["results_dir"] = str(tmp_path)
    monkeypatch.setattr(news, "RAW_DIR", str(tmp_path / "no-raw-headlines"))
    monkeypatch.setattr(backtest, "resolve_path", lambda p: Path(p))

    out = backtest.news_daily_features(cfg, "TCS.NS", days)
    assert list(out.columns) == NEWS_COLUMNS
    assert out.iloc[:4].to_numpy().min() == 0.5
    assert out.iloc[4:].to_numpy().sum() == 0  # days after the saved range get no news
