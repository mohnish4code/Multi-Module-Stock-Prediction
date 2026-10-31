import { useCallback, useEffect, useState } from "react";

import { api } from "./api";
import { AnalysisView, ErrorBox } from "./components/AnalysisView";
import { BacktestView } from "./components/BacktestView";
import { Sidebar, TickerTape, TopBar } from "./components/Layout";
import { Leaderboard } from "./components/Leaderboard";
import { signed, signedPct, tone } from "./format";
import type { Quote } from "./types";
import { useApi } from "./useApi";

type Tab = "analysis" | "backtest" | "leaderboard";
const TABS: { id: Tab; label: string }[] = [
  { id: "analysis", label: "Analysis" },
  { id: "backtest", label: "Backtest" },
  { id: "leaderboard", label: "Leaderboard" },
];

function readUrl(): { ticker: string | null; tab: Tab } {
  const params = new URLSearchParams(window.location.search);
  const tab = params.get("tab") as Tab | null;
  return { ticker: params.get("ticker"), tab: tab && TABS.some((t) => t.id === tab) ? tab : "analysis" };
}

function storedTheme(): string {
  try {
    return localStorage.getItem("theme") ?? "dark";
  } catch {
    return "dark";
  }
}

export default function App() {
  const initial = readUrl();
  const [tab, setTab] = useState<Tab>(initial.tab);
  const [selected, setSelected] = useState<string | null>(initial.ticker);
  const [theme, setTheme] = useState(storedTheme);
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});

  const tickers = useApi((s) => api.tickers(s), "tickers");
  const health = useApi((s) => api.health(s), "health");
  const ticker = selected ?? tickers.data?.[0]?.ticker ?? null;
  const backtest = useApi((s) => api.backtest(ticker!, s), ticker);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // storage unavailable (private mode); theme still applies for this visit
    }
  }, [theme]);

  useEffect(() => {
    const load = () => api.quotes().then(setQuotes).catch(() => undefined);
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!ticker) return;
    const params = new URLSearchParams({ ticker, tab });
    window.history.replaceState(null, "", `?${params}`);
  }, [ticker, tab]);

  const select = useCallback(
    (value: string) => {
      const match = tickers.data?.find((t) => t.ticker === value || t.folder === value);
      if (!match) return;
      setSelected(match.ticker);
      if (tab === "leaderboard") setTab("analysis");
    },
    [tickers.data, tab],
  );

  if (tickers.error) {
    return (
      <div className="fullscreen">
        <ErrorBox message={`${tickers.error} Start it with: uvicorn src.api.main:app --port 8000`} onRetry={tickers.reload} />
      </div>
    );
  }

  const info = tickers.data?.find((t) => t.ticker === ticker);
  const quote = ticker ? quotes[ticker] : undefined;

  return (
    <div className="app">
      <TopBar health={health.data} theme={theme} onTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />
      {tickers.data && <TickerTape tickers={tickers.data} quotes={quotes} onSelect={select} />}
      <div className="body">
        {tickers.data && ticker && (
          <Sidebar tickers={tickers.data} quotes={quotes} selected={ticker} onSelect={select} />
        )}
        <main className="main">
          {info && (
            <div className="instrument-head">
              <div>
                <h1>{info.folder}</h1>
                <span className="muted">{info.name}</span>
              </div>
              {quote && (
                <div className="head-quote">
                  <span className="mono big">{quote.price.toLocaleString("en-IN", { minimumFractionDigits: 2 })}</span>
                  <span className={`mono ${tone(quote.change)}`}>
                    {signed(quote.change)} ({signedPct(quote.change_pct)})
                  </span>
                  <small className="muted">last close</small>
                </div>
              )}
            </div>
          )}
          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t.id} className={t.id === tab ? "active" : ""} onClick={() => setTab(t.id)}>{t.label}</button>
            ))}
          </nav>

          {tab === "analysis" && ticker && <AnalysisView ticker={ticker} backtest={backtest.data} theme={theme} />}
          {tab === "backtest" && (backtest.error ? <ErrorBox message={backtest.error} onRetry={backtest.reload} /> :
            backtest.data ? <BacktestView key={ticker} data={backtest.data} theme={theme} /> :
              <div className="skeleton" style={{ height: 480 }} />)}
          {tab === "leaderboard" && <Leaderboard onSelect={select} />}

          <footer className="footer">
            AlphaStream is a research project. Forecasts are model output, not investment advice.
          </footer>
        </main>
      </div>
    </div>
  );
}
