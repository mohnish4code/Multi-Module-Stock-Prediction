const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 });

export const price = (v: number) => `₹${inr.format(v)}`;
export const num = (v: number | null | undefined, digits = 2) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(digits));
export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
export const signedPct = (v: number, digits = 2) => `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
export const signed = (v: number, digits = 2) => `${v > 0 ? "+" : ""}${inr.format(Number(v.toFixed(digits)))}`;

export function tone(value: number | string | null | undefined): "up" | "down" | "flat" {
  if (typeof value === "number") return value > 0 ? "up" : value < 0 ? "down" : "flat";
  if (value == null) return "flat";
  if (["BULLISH", "BUY", "POSITIVE"].includes(value)) return "up";
  if (["BEARISH", "SELL", "NEGATIVE"].includes(value)) return "down";
  return "flat";
}

/** Next weekday after an ISO date (NSE holidays are ignored). */
export function nextBusinessDay(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  do {
    d.setUTCDate(d.getUTCDate() + 1);
  } while (d.getUTCDay() === 0 || d.getUTCDay() === 6);
  return d.toISOString().slice(0, 10);
}

export function timeAgo(iso: string | null, now = Date.now()): string {
  if (!iso) return "date unknown";
  const hours = (now - new Date(iso).getTime()) / 3_600_000;
  if (hours < 1) return "just now";
  if (hours < 24) return `${Math.floor(hours)}h ago`;
  const days = Math.floor(hours / 24);
  return days === 1 ? "1 day ago" : `${days} days ago`;
}

export const MODEL_LABELS: Record<string, string> = {
  buy_and_hold: "Buy & hold",
  random_walk: "Random walk",
  persistence: "Persistence",
  majority: "Majority",
  arima: "ARIMA",
  xgboost: "XGBoost",
  lstm: "LSTM",
  transformer: "Transformer",
  "xgboost+news": "XGBoost + news",
  "lstm+news": "LSTM + news",
};
export const modelLabel = (m: string) => MODEL_LABELS[m] ?? m;
export const BASELINES = new Set(["random_walk", "persistence", "majority", "buy_and_hold"]);

export const MODEL_COLORS: Record<string, string> = {
  buy_and_hold: "#e2e8f0",
  "xgboost+news": "#22c55e",
  xgboost: "#3b82f6",
  lstm: "#a855f7",
  "lstm+news": "#ec4899",
  transformer: "#f59e0b",
  arima: "#06b6d4",
  majority: "#64748b",
  persistence: "#ef4444",
};
