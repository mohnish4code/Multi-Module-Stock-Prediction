import type { Backtest, Health, MetricRow, Prediction, Quote, TickerInfo } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError("Cannot reach the API. Is the server running?", 0);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // body was not JSON
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: (signal?: AbortSignal) => get<Health>("/health", signal),
  tickers: (signal?: AbortSignal) => get<TickerInfo[]>("/tickers", signal),
  quotes: (signal?: AbortSignal) => get<Record<string, Quote>>("/quotes", signal),
  predict: (ticker: string, signal?: AbortSignal) => get<Prediction>(`/predict/${ticker}`, signal),
  backtest: (ticker: string, signal?: AbortSignal) => get<Backtest>(`/backtest/${ticker}`, signal),
  summary: (signal?: AbortSignal) => get<MetricRow[]>("/summary", signal),
};
