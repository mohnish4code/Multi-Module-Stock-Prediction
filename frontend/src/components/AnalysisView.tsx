import { useMemo, useRef, useState } from "react";

import { api } from "../api";
import { num, price, signed, signedPct, timeAgo, tone } from "../format";
import type { Prediction, Sentiment } from "../types";
import { useApi } from "../useApi";
import { PriceChart } from "./PriceChart";

// Predictions are slow on a cold server, so keep them for the session.
const cache = new Map<string, { at: number; data: Prediction }>();
const MAX_AGE_MS = 10 * 60_000;

export function AnalysisView({ ticker, theme }: { ticker: string; theme: string }) {
  const [force, setForce] = useState(0);
  const bypassCache = useRef(false);
  const refresh = () => {
    bypassCache.current = true;
    setForce((f) => f + 1);
  };
  const { data, error, loading } = useApi(async (signal) => {
    const hit = cache.get(ticker);
    const bypass = bypassCache.current;
    bypassCache.current = false;
    if (hit && !bypass && Date.now() - hit.at < MAX_AGE_MS) return hit.data;
    const fresh = await api.predict(ticker, signal);
    cache.set(ticker, { at: Date.now(), data: fresh });
    return fresh;
  }, `${ticker}:${force}`);

  if (error) return <ErrorBox message={error} onRetry={refresh} />;
  if (!data || loading) return <AnalysisSkeleton />;

  return (
    <div className="stack">
      <SignalBanner data={data} onRefresh={refresh} />
      <Kpis data={data} />
      <PriceChart history={data.history} forecast={data.forecast} theme={theme} />
      <div className="grid-3">
        <ModelCard data={data} />
        <TechnicalCard data={data} />
        <NewsCard data={data} />
      </div>
      <DecisionPanel data={data} />
      <NewsFeed data={data} />
    </div>
  );
}

function SignalBanner({ data, onRefresh }: { data: Prediction; onRefresh: () => void }) {
  const d = data.decision;
  return (
    <div className="banner">
      <span className={`rec rec-${tone(d.recommendation)}`}>{d.recommendation}</span>
      <span className="muted">
        coordinated <b className={tone(d.coordinated_signal)}>{d.coordinated_signal}</b> · {d.agreement.toLowerCase()} agreement
        · data as of {data.forecast.as_of}
      </span>
      <button className="ghost-btn" onClick={onRefresh}>↻ Refresh</button>
    </div>
  );
}

function Kpis({ data }: { data: Prediction }) {
  const f = data.forecast;
  const d = data.decision;
  const n = data.news;
  return (
    <div className="kpis">
      <Kpi label="Predicted close" value={price(f.predicted_price)} sub={`${f.model} · next session`} />
      <Kpi label="Expected move" value={signedPct(f.predicted_change_pct)} sub={`${signed(f.predicted_change)} vs last close`}
        cls={tone(f.predicted_change)} />
      <Kpi label="News sentiment" value={n.sentiment_label} sub={`score ${num(n.sentiment_score)} · ${n.total_news} headlines`}
        cls={tone(n.sentiment_label)} />
      <Kpi label="Risk" value={d.risk_level} sub={`score ${d.risk_score}/10`}
        cls={d.risk_level === "HIGH" ? "down" : d.risk_level === "LOW" ? "up" : "flat"} />
    </div>
  );
}

function Kpi({ label, value, sub, cls = "" }: { label: string; value: string; sub: string; cls?: string }) {
  return (
    <div className="card kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value mono ${cls}`}>{value}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}

function Row({ k, v, cls = "" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="row">
      <span className="muted">{k}</span>
      <span className={`mono ${cls}`}>{v}</span>
    </div>
  );
}

function ModelCard({ data }: { data: Prediction }) {
  const f = data.forecast;
  return (
    <div className="card">
      <div className="card-title">1 · Model forecast</div>
      <span className={`badge ${tone(f.direction)}`}>{f.direction}</span>
      <Row k="Model" v={f.model} />
      <Row k="Predicted return" v={`${(f.predicted_return * 100).toFixed(3)}%`} cls={tone(f.predicted_return)} />
      <Row k="Predicted close" v={price(f.predicted_price)} />
      <Row k="Trained" v={f.model_trained_at.slice(0, 10)} />
    </div>
  );
}

function TechnicalCard({ data }: { data: Prediction }) {
  const t = data.technical;
  return (
    <div className="card">
      <div className="card-title">2 · Technical analysis</div>
      <span className={`badge ${tone(t.technical_signal)}`}>{t.technical_signal}</span>
      <div className="row">
        <span className="muted">RSI (14)</span>
        <span className="rsi">
          <span className="rsi-track"><span className="rsi-fill" style={{ left: `${Math.min(100, Math.max(0, t.rsi))}%` }} /></span>
          <span className="mono">{num(t.rsi, 1)}</span>
        </span>
      </div>
      <Row k="Trend (SMA 20/50)" v={t.trend} cls={tone(t.trend)} />
      <Row k="MACD" v={t.macd_trend} cls={tone(t.macd_trend)} />
      <Row k="Volatility" v={t.volatility_level} />
    </div>
  );
}

function NewsCard({ data }: { data: Prediction }) {
  const n = data.news;
  const total = Math.max(1, n.total_news);
  return (
    <div className="card">
      <div className="card-title">3 · News sentiment ({n.method})</div>
      <span className={`badge ${tone(n.sentiment_label)}`}>{n.sentiment_label}</span>
      <div className="split-bar" title={`${n.positive} positive · ${n.neutral} neutral · ${n.negative} negative`}>
        <span className="up-bg" style={{ width: `${(n.positive / total) * 100}%` }} />
        <span className="flat-bg" style={{ width: `${(n.neutral / total) * 100}%` }} />
        <span className="down-bg" style={{ width: `${(n.negative / total) * 100}%` }} />
      </div>
      <Row k="Headlines" v={`${n.positive} / ${n.neutral} / ${n.negative}`} />
      <Row k="Score" v={num(n.sentiment_score)} cls={tone(n.sentiment_score)} />
      <Row k="Most recent" v={timeAgo(n.most_recent)} />
    </div>
  );
}

function Gauge({ value, label }: { value: number; label: string }) {
  const angle = Math.PI * (1 - value / 100);
  const x = 60 + 48 * Math.cos(angle);
  const y = 60 - 48 * Math.sin(angle);
  return (
    <svg viewBox="0 0 120 70" className="gauge" role="img" aria-label={`${label} ${value} of 100`}>
      <path d="M12 60 A48 48 0 0 1 108 60" className="gauge-track" />
      <path d={`M12 60 A48 48 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}`} className="gauge-fill" />
      <text x="60" y="56" textAnchor="middle" className="gauge-text">{value}</text>
    </svg>
  );
}

function DecisionPanel({ data }: { data: Prediction }) {
  const d = data.decision;
  const votes = [
    { name: "Model", value: data.forecast.direction },
    { name: "Technical", value: data.technical.technical_signal },
    { name: "News", value: data.news.sentiment_label },
  ];
  return (
    <div className="decision">
      <div className="card">
        <div className="card-title">Module votes</div>
        <div className="votes">
          {votes.map((v) => (
            <div key={v.name} className={`vote vote-${tone(v.value)}`}>
              <small>{v.name}</small>
              <b>{v.value}</b>
            </div>
          ))}
        </div>
        <div className="gauges">
          <div>
            <Gauge value={d.confidence_score} label="Agreement" />
            <small className="muted">Agreement score · {d.confidence_level}</small>
          </div>
          <div>
            <div className="risk-meter">
              {Array.from({ length: 10 }, (_, i) => (
                <span key={i} className={i < d.risk_score ? `on risk-${d.risk_level.toLowerCase()}` : ""} />
              ))}
            </div>
            <small className="muted">Risk {d.risk_score}/10 · {d.risk_level}</small>
          </div>
        </div>
      </div>
      <div className={`card decision-card border-${tone(d.recommendation)}`}>
        <div className="card-title">Final recommendation</div>
        <div className={`rec-big ${tone(d.recommendation)}`}>{d.recommendation}</div>
        <p>{d.explanation}</p>
        {d.risk_factors.length > 0 && (
          <ul className="factors">{d.risk_factors.map((r) => <li key={r}>{r}</li>)}</ul>
        )}
        <p className="fineprint">
          Agreement and risk scores are rule-based heuristics, not calibrated probabilities. See the Backtest tab
          for how the forecast has actually performed.
        </p>
      </div>
    </div>
  );
}

function NewsFeed({ data }: { data: Prediction }) {
  const [filter, setFilter] = useState<Sentiment | "ALL">("ALL");
  const items = useMemo(
    () => data.news.news.filter((h) => filter === "ALL" || h.sentiment === filter),
    [data.news.news, filter],
  );
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>News feed</h3>
        <div className="segmented">
          {(["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE"] as const).map((f) => (
            <button key={f} className={f === filter ? "active" : ""} onClick={() => setFilter(f)}>
              {f === "ALL" ? "All" : f[0] + f.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
      </div>
      <ul className="news">
        {items.map((h) => (
          <li key={h.title} className={`news-item border-${tone(h.sentiment)}`}>
            <a href={h.link || undefined} target="_blank" rel="noreferrer">{h.title}</a>
            <div className="news-meta">
              <span>{h.publisher}</span>
              <span>{timeAgo(h.published)}</span>
              <span className={`badge-sm ${tone(h.sentiment)}`}>{h.sentiment}</span>
              <span className="score-bar" title={`score ${h.score.toFixed(2)}`}>
                <span className={tone(h.score) + "-bg"} style={{ width: `${Math.min(100, Math.abs(h.score) * 100)}%` }} />
              </span>
            </div>
          </li>
        ))}
        {items.length === 0 && <li className="muted empty">No headlines for this filter.</li>}
      </ul>
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="card error-box">
      <b>Something went wrong</b>
      <p>{message}</p>
      {onRetry && <button className="primary-btn" onClick={onRetry}>Try again</button>}
    </div>
  );
}

function AnalysisSkeleton() {
  return (
    <div className="stack">
      <div className="skeleton" style={{ height: 44 }} />
      <div className="kpis">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 96 }} />)}</div>
      <div className="skeleton" style={{ height: 420 }} />
      <p className="muted center">Running the model, technical read and FinBERT news scoring…</p>
    </div>
  );
}
