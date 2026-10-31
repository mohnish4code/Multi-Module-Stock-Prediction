"""Response models. Field names match the dicts returned by src.pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]
Sentiment = Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]


class Forecast(BaseModel):
    model: str
    model_trained_at: str
    as_of: str = Field(description="Date of the last bar the forecast is made from")
    current_price: float
    predicted_price: float
    predicted_return: float = Field(description="Predicted next-day log return")
    predicted_change: float
    predicted_change_pct: float
    direction: Direction


class Technical(BaseModel):
    rsi: float
    rsi_signal: Direction
    sma_20: float
    sma_50: float
    trend: Direction
    macd: float
    macd_signal: float
    macd_trend: Direction
    volatility: float
    volatility_level: Literal["LOW", "MEDIUM", "HIGH"]
    technical_signal: Direction
    bullish_signals: int
    bearish_signals: int
    neutral_signals: int
    signals: list[str]


class Headline(BaseModel):
    title: str
    publisher: str
    link: str
    published: str | None
    age_days: float | None
    sentiment: Sentiment
    score: float


class News(BaseModel):
    method: Literal["finbert", "keyword"]
    total_news: int
    positive: int
    neutral: int
    negative: int
    sentiment_score: float
    sentiment_label: Sentiment
    most_recent: str | None
    newest_age_days: float | None
    fetched_at: str | None = None
    news: list[Headline]


class Decision(BaseModel):
    coordinated_signal: Direction
    agreement: Literal["STRONG", "MODERATE", "MIXED"]
    bullish_modules: int
    bearish_modules: int
    neutral_modules: int
    confidence_score: int = Field(description="Rule-based agreement score, not a calibrated probability")
    confidence_level: Literal["HIGH", "MODERATE", "LOW"]
    risk_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_factors: list[str]
    recommendation: Literal["BUY", "HOLD", "SELL"]
    explanation: str


class Bar(BaseModel):
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    sma_20: float | None
    ema_20: float | None
    bb_upper: float | None
    bb_lower: float | None


class Prediction(BaseModel):
    ticker: str
    name: str
    forecast: Forecast
    technical: Technical
    news: News
    decision: Decision
    history: list[Bar]


class BacktestRow(BaseModel):
    """One row of results/metrics.csv (fold is 0..n-1 or 'all')."""

    model_config = ConfigDict(extra="allow")
    ticker: str
    model: str
    fold: str
    test_start: str
    test_end: str
    n: int
    dir_acc: float | None
    p_vs_50: float | None
    mape: float | None
    rmse: float | None
    strat_sharpe: float | None
    strat_cagr: float | None
    strat_max_dd: float | None
    bh_sharpe: float | None
    bh_cagr: float | None
    exposure: float | None


class Equity(BaseModel):
    dates: list[str]
    curves: dict[str, list[float]]


class Backtest(BaseModel):
    ticker: str
    cost_per_side: float
    metrics: list[BacktestRow]
    equity: Equity


class TickerInfo(BaseModel):
    ticker: str
    name: str
    folder: str
    model_ready: bool


class Quote(BaseModel):
    price: float
    prev: float
    change: float
    change_pct: float


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    serving_model: str
    models_ready: int
    models_expected: int
    data_end: str | None


class ErrorBody(BaseModel):
    error: str
    detail: str
