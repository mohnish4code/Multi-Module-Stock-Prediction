import { useEffect, useMemo, useState } from "react";

import { signedPct, tone } from "../format";
import type { Health, Quote, TickerInfo } from "../types";

function istNow() {
  return new Date(Date.now() + (330 + new Date().getTimezoneOffset()) * 60_000);
}

function marketOpen(d: Date) {
  const minutes = d.getHours() * 60 + d.getMinutes();
  return d.getDay() > 0 && d.getDay() < 6 && minutes >= 555 && minutes <= 930;
}

export function TopBar({ health, theme, onTheme }: { health: Health | null; theme: string; onTheme: () => void }) {
  const [now, setNow] = useState(istNow());
  useEffect(() => {
    const id = setInterval(() => setNow(istNow()), 1000);
    return () => clearInterval(id);
  }, []);
  const open = marketOpen(now);

  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo">◆</span> AlphaStream
        <span className="brand-sub">NSE forecasting terminal</span>
      </div>
      <div className="topbar-right">
        {health && (
          <span className={`pill ${health.status === "ok" ? "pill-up" : "pill-warn"}`} title="API status">
            {health.serving_model} · {health.models_ready}/{health.models_expected} models
          </span>
        )}
        <span className="clock mono">
          <span className={`dot ${open ? "dot-up" : "dot-down"}`} /> NSE {open ? "OPEN" : "CLOSED"} ·{" "}
          {now.toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" })} IST
        </span>
        <button className="icon-btn" onClick={onTheme} title="Toggle theme" aria-label="Toggle theme">
          {theme === "dark" ? "☀" : "☾"}
        </button>
      </div>
    </header>
  );
}

export function TickerTape({ tickers, quotes, onSelect }: {
  tickers: TickerInfo[]; quotes: Record<string, Quote>; onSelect: (t: string) => void;
}) {
  return (
    <div className="tape">
      {tickers.map((t) => {
        const q = quotes[t.ticker];
        return (
          <button key={t.ticker} className="tape-item mono" onClick={() => onSelect(t.ticker)}>
            <b>{t.folder}</b>
            {q ? (
              <>
                <span className="muted">{q.price.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span>
                <span className={tone(q.change_pct)}>{q.change_pct >= 0 ? "▲" : "▼"} {signedPct(q.change_pct)}</span>
              </>
            ) : (
              <span className="muted">–</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function Sidebar({ tickers, quotes, selected, onSelect }: {
  tickers: TickerInfo[]; quotes: Record<string, Quote>; selected: string; onSelect: (t: string) => void;
}) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? tickers.filter((t) => `${t.folder} ${t.name}`.toLowerCase().includes(q)) : tickers;
  }, [tickers, query]);

  return (
    <aside className="sidebar">
      <input className="search" placeholder="Search stocks…" value={query} onChange={(e) => setQuery(e.target.value)} />
      <ul className="instruments">
        {shown.map((t) => {
          const q = quotes[t.ticker];
          return (
            <li key={t.ticker}>
              <button className={`instrument ${t.ticker === selected ? "selected" : ""}`} onClick={() => onSelect(t.ticker)}>
                <span>
                  <b>{t.folder}</b>
                  <small>{t.name}</small>
                </span>
                {q && (
                  <span className="instrument-quote mono">
                    {q.price.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    <small className={tone(q.change_pct)}>{signedPct(q.change_pct)}</small>
                  </span>
                )}
              </button>
            </li>
          );
        })}
        {shown.length === 0 && <li className="muted empty">No match</li>}
      </ul>
      <p className="sidebar-note">
        Model forecast · technical rules · FinBERT news → one decision. Data: Yahoo Finance EOD, Google News.
        Horizon: next trading day.
      </p>
    </aside>
  );
}
