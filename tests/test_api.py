from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import pipeline
from src.api import main as api
from tests.conftest import make_ohlcv


@pytest.fixture
def stub_analysis(cfg: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A real analyse() result from synthetic data and a trivially fitted model."""
    import numpy as np

    from src.models.baselines import Majority
    from src.models.dataset import Dataset

    ohlcv = make_ohlcv()
    ds = Dataset.from_frame("TCS.NS", pipeline.build_feature_frame(ohlcv))
    out = tmp_path / "TCS" / "majority"
    Majority().fit(ds, np.arange(59, 600), np.arange(610, 700)).save(out)
    (out / "manifest.json").write_text(json.dumps({"model": "majority", "trained_at": "2026-01-01T00:00",
                                                   "data_end": "2026-01-01"}))
    local = {**cfg, "serving": {**cfg["serving"], "model": "majority", "feature_set": "base",
                                     "models_dir": str(tmp_path)}}
    with monkeypatch.context() as m:
        m.setattr(pipeline, "resolve_path", Path)
        pipeline._cached_model.cache_clear()
        result = pipeline.analyse("TCS.NS", local, include_news=False, ohlcv=ohlcv)
    pipeline._cached_model.cache_clear()
    return result


@pytest.fixture
def client(cfg: dict[str, Any], stub_analysis: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
           tmp_path: Path) -> TestClient:
    calls: list[tuple[str, bool]] = []

    def fake_analyse(ticker: str, _cfg: dict, include_news: bool = True) -> dict[str, Any]:
        calls.append((ticker, include_news))
        return {**stub_analysis, "ticker": ticker}

    def fake_backtest(ticker: str, _cfg: dict) -> dict[str, Any]:
        row = {"ticker": "TCS", "model": "xgboost", "fold": "all", "test_start": "2020-01-01",
               "test_end": "2026-01-01", "n": 1512, "dir_acc": 50.4, "p_vs_50": 0.4, "mape": 1.06, "rmse": 21.0,
               "strat_sharpe": 0.4, "strat_cagr": 0.08, "strat_max_dd": -0.3, "bh_sharpe": 0.6, "bh_cagr": 0.14,
               "exposure": 0.9, "excess_cagr_vs_bh": -0.06}
        return {"ticker": ticker, "cost_per_side": 0.0012, "metrics": [row],
                "equity": {"dates": ["2020-01-01"], "curves": {"xgboost": [1.0], "buy_and_hold": [1.0]}}}

    monkeypatch.setattr(pipeline, "analyse", fake_analyse)
    monkeypatch.setattr(pipeline, "backtest_summary", fake_backtest)
    monkeypatch.setattr(api, "get_quotes", lambda tickers: {"TCS.NS": {"price": 1.0, "prev": 1.0, "change": 0.0,
                                                                        "change_pct": 0.0}})
    cfg["serving"]["models_dir"] = str(tmp_path / "empty")
    test_client = TestClient(api.create_app(cfg, rate_limit="5/minute"))
    test_client.calls = calls
    return test_client


def test_predict_returns_validated_payload(client: TestClient) -> None:
    r = client.get("/predict/tcs?include_news=false")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "TCS.NS"
    assert body["decision"]["recommendation"] in ("BUY", "HOLD", "SELL")
    assert len(body["history"]) == 90
    assert client.calls == [("TCS.NS", False)]


def test_unknown_ticker_is_404(client: TestClient) -> None:
    r = client.get("/predict/AAPL")
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_ticker"


def test_invalid_ticker_is_422(client: TestClient) -> None:
    assert client.get("/predict/bad$ticker").status_code == 422


def test_backtest(client: TestClient) -> None:
    r = client.get("/backtest/INFY")
    assert r.status_code == 200
    assert r.json()["metrics"][0]["excess_cagr_vs_bh"] == -0.06  # extra columns pass through


def test_health_reports_missing_models(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["models_ready"] == 0
    assert body["models_expected"] == 10


def test_missing_model_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def not_trained(*_args: Any, **_kwargs: Any) -> dict:
        raise FileNotFoundError("No trained 'xgboost' model for TCS")

    monkeypatch.setattr(pipeline, "analyse", not_trained)
    r = client.get("/predict/TCS")
    assert r.status_code == 503
    assert r.json()["error"] == "not_ready"


def test_rate_limit(client: TestClient) -> None:
    codes = [client.get("/tickers").status_code for _ in range(7)]
    assert codes[:5] == [200] * 5
    assert codes[-1] == 429
    assert client.get("/health").status_code == 200  # health is exempt


def test_quotes(client: TestClient) -> None:
    assert client.get("/quotes").json()["TCS.NS"]["price"] == 1.0


def test_tickers(client: TestClient) -> None:
    body = client.get("/tickers").json()
    assert len(body) == 10
    assert not any(t["model_ready"] for t in body)
