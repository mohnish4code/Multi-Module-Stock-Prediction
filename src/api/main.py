"""REST API over the shared pipeline.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints: /health, /tickers, /quotes, /predict/{ticker}, /backtest/{ticker}.
Docs at /docs. Rate limit per client IP comes from RATE_LIMIT (default 30/minute).
WARMUP=1 pre-runs every ticker in the background at startup (loads FinBERT, fills the caches).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src import pipeline
from src.api.schemas import Backtest, ErrorBody, Health, Prediction, Quote, TickerInfo
from src.config import load_config, resolve_path, resolve_ticker
from src.data.loader import get_quotes
from src.logging_setup import setup_logging

logger = logging.getLogger(__name__)

VERSION = "2.0.0"
TICKER_PATH = Path(..., description="NSE symbol, e.g. TCS, TCS.NS or INFY", min_length=1, max_length=20,
                   pattern=r"^[A-Za-z0-9&.\-]+$")
ERRORS = {404: {"model": ErrorBody}, 422: {"model": ErrorBody}, 429: {"model": ErrorBody},
          503: {"model": ErrorBody}}


class UnknownTicker(LookupError):
    pass


def _manifest(cfg: dict[str, Any], folder: str) -> dict[str, Any] | None:
    path = resolve_path(cfg["serving"]["models_dir"]) / folder / pipeline.serving_label(cfg) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def warm_up(cfg: dict[str, Any]) -> None:
    """Loads FinBERT and runs every ticker once so the first real requests hit warm caches."""
    started = time.perf_counter()
    for ticker in cfg["tickers"]:
        try:
            pipeline.analyse(ticker, cfg)
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            logger.warning("Warm-up for %s failed: %s", ticker, exc)
    logger.info("Warm-up finished in %.0fs", time.perf_counter() - started)


def create_app(cfg: dict[str, Any] | None = None, rate_limit: str | None = None) -> FastAPI:
    cfg = cfg or load_config()
    limit = rate_limit or os.getenv("RATE_LIMIT", "30/minute")
    limiter = Limiter(key_func=get_remote_address, default_limits=[limit])

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        ready = sum(_manifest(cfg, i["folder"]) is not None for i in cfg["tickers"].values())
        logger.info("API starting: %d/%d %s models available", ready, len(cfg["tickers"]), pipeline.serving_label(cfg))
        if os.getenv("WARMUP", "0").lower() in ("1", "true", "yes"):
            threading.Thread(target=warm_up, args=(cfg,), daemon=True, name="warm-up").start()
        yield

    app = FastAPI(title="AlphaStream API", version=VERSION, lifespan=lifespan,
                  description="Next-day NSE forecasts, signals and walk-forward backtest results.")
    app.state.limiter = limiter

    def error(status: int, kind: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": kind, "detail": detail})

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(_: Request, exc: RateLimitExceeded) -> JSONResponse:
        return error(429, "rate_limited", f"Rate limit exceeded: {exc.detail}")

    @app.exception_handler(UnknownTicker)
    async def _unknown(_: Request, exc: UnknownTicker) -> JSONResponse:
        return error(404, "unknown_ticker", str(exc))

    @app.exception_handler(FileNotFoundError)
    async def _not_ready(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return error(503, "not_ready", str(exc))

    @app.exception_handler(ValueError)
    async def _bad_data(_: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Data error: %s", exc)
        return error(422, "data_error", str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return error(500, "internal_error", "Unexpected server error")

    def ticker_or_404(value: str) -> str:
        try:
            return resolve_ticker(cfg, value)
        except KeyError as exc:
            raise UnknownTicker(f"Unknown ticker '{value}'. Known: {', '.join(cfg['tickers'])}") from exc

    @app.get("/health", response_model=Health)
    @limiter.exempt
    def health() -> Health:
        manifests = [_manifest(cfg, i["folder"]) for i in cfg["tickers"].values()]
        ready = [m for m in manifests if m]
        return Health(
            status="ok" if len(ready) == len(manifests) else "degraded",
            version=VERSION,
            serving_model=pipeline.serving_label(cfg),
            models_ready=len(ready),
            models_expected=len(manifests),
            data_end=max((m["data_end"] for m in ready), default=None),
        )

    @app.get("/tickers", response_model=list[TickerInfo])
    @limiter.limit(limit)
    def tickers(request: Request) -> list[TickerInfo]:
        return [
            TickerInfo(ticker=t, name=info["name"], folder=info["folder"],
                       model_ready=_manifest(cfg, info["folder"]) is not None)
            for t, info in cfg["tickers"].items()
        ]

    @app.get("/quotes", response_model=dict[str, Quote])
    @limiter.limit(limit)
    def quotes(request: Request) -> dict[str, Any]:
        return get_quotes(list(cfg["tickers"]))

    @app.get("/predict/{ticker}", response_model=Prediction, responses=ERRORS)
    @limiter.limit(limit)
    def predict(
        request: Request,
        ticker: str = TICKER_PATH,
        include_news: bool = Query(True, description="Fetch and score live headlines (slower)"),
    ) -> dict[str, Any]:
        return pipeline.analyse(ticker_or_404(ticker), cfg, include_news=include_news)

    @app.get("/backtest/{ticker}", response_model=Backtest, responses=ERRORS)
    @limiter.limit(limit)
    def backtest(request: Request, ticker: str = TICKER_PATH) -> dict[str, Any]:
        return pipeline.backtest_summary(ticker_or_404(ticker), cfg)

    return app


app = create_app()
