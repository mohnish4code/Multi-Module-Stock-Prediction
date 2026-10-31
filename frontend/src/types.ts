// Mirrors src/api/schemas.py.

export type Direction = "BULLISH" | "BEARISH" | "NEUTRAL";
export type Sentiment = "POSITIVE" | "NEUTRAL" | "NEGATIVE";
export type Recommendation = "BUY" | "HOLD" | "SELL";

export interface TickerInfo {
  ticker: string;
  name: string;
  folder: string;
  model_ready: boolean;
}

export interface Quote {
  price: number;
  prev: number;
  change: number;
  change_pct: number;
}

export interface Forecast {
  model: string;
  model_trained_at: string;
  as_of: string;
  current_price: number;
  predicted_price: number;
  predicted_return: number;
  predicted_change: number;
  predicted_change_pct: number;
  direction: Direction;
}

export interface Technical {
  rsi: number;
  rsi_signal: Direction;
  sma_20: number;
  sma_50: number;
  trend: Direction;
  macd: number;
  macd_signal: number;
  macd_trend: Direction;
  volatility: number;
  volatility_level: "LOW" | "MEDIUM" | "HIGH";
  technical_signal: Direction;
  signals: string[];
}

export interface Headline {
  title: string;
  publisher: string;
  link: string;
  published: string | null;
  age_days: number | null;
  sentiment: Sentiment;
  score: number;
}

export interface News {
  method: "finbert" | "keyword";
  total_news: number;
  positive: number;
  neutral: number;
  negative: number;
  sentiment_score: number;
  sentiment_label: Sentiment;
  most_recent: string | null;
  fetched_at: string | null;
  news: Headline[];
}

export interface Decision {
  coordinated_signal: Direction;
  agreement: "STRONG" | "MODERATE" | "MIXED";
  bullish_modules: number;
  bearish_modules: number;
  neutral_modules: number;
  confidence_score: number;
  confidence_level: "HIGH" | "MODERATE" | "LOW";
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  risk_factors: string[];
  recommendation: Recommendation;
  explanation: string;
}

export interface Bar {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  sma_20: number | null;
  ema_20: number | null;
  bb_upper: number | null;
  bb_lower: number | null;
}

export interface Prediction {
  ticker: string;
  name: string;
  forecast: Forecast;
  technical: Technical;
  news: News;
  decision: Decision;
  history: Bar[];
}

export interface MetricRow {
  ticker: string;
  model: string;
  fold: string;
  test_start: string;
  test_end: string;
  n: number;
  dir_acc: number | null;
  p_vs_50: number | null;
  mape: number | null;
  rmse: number | null;
  strat_sharpe: number | null;
  strat_cagr: number | null;
  strat_max_dd: number | null;
  bh_sharpe: number | null;
  bh_cagr: number | null;
  bh_max_dd?: number | null;
  strat_total_return?: number | null;
  bh_total_return?: number | null;
  exposure: number | null;
  excess_cagr_vs_bh?: number | null;
}

export interface Backtest {
  ticker: string;
  cost_per_side: number;
  metrics: MetricRow[];
  equity: { dates: string[]; curves: Record<string, number[]> };
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  serving_model: string;
  models_ready: number;
  models_expected: number;
  data_end: string | null;
}
