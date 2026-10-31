"""HTTP client for the AlphaStream API. The UI reads everything through here."""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
_http = httpx.Client(base_url=API_URL, timeout=httpx.Timeout(180.0, connect=5.0))


class ClientError(RuntimeError):
    """Raised with a user-facing message when the API is down or returns an error."""


def _get(path: str, **params: Any) -> Any:
    try:
        response = _http.get(path, params=params or None)
    except httpx.ConnectError as exc:
        raise ClientError(f"Cannot reach the API at {API_URL}. Is it running?") from exc
    except httpx.TimeoutException as exc:
        raise ClientError("The API took too long to respond.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ClientError(f"API error {response.status_code}: {detail}")
    return response.json()


def tickers() -> dict[str, dict[str, Any]]:
    return {t["ticker"]: t for t in _get("/tickers")}


def quotes(symbols: list[str]) -> dict[str, dict[str, float]]:
    return _get("/quotes")


def analysis(ticker: str) -> dict[str, Any]:
    return _get(f"/predict/{ticker}")


def backtest(ticker: str) -> dict[str, Any]:
    return _get(f"/backtest/{ticker}")
