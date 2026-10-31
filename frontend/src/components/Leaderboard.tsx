import { useMemo, useState } from "react";

import { api } from "../api";
import { modelLabel } from "../format";
import type { MetricRow } from "../types";
import { useApi } from "../useApi";
import { ErrorBox } from "./AnalysisView";

const METRICS = {
  dir_acc: { label: "Direction accuracy", centre: 50, span: 4, fmt: (v: number) => `${v.toFixed(1)}%` },
  strat_sharpe: { label: "Sharpe (net)", centre: 0, span: 1, fmt: (v: number) => v.toFixed(2) },
  excess_cagr_vs_bh: { label: "CAGR minus buy & hold", centre: 0, span: 0.3, fmt: (v: number) => `${(v * 100).toFixed(1)}%` },
} as const;
type Metric = keyof typeof METRICS;

const MODELS = ["majority", "arima", "xgboost", "lstm", "transformer", "xgboost+news", "lstm+news"];

function heat(v: number, centre: number, span: number): string {
  const t = Math.max(-1, Math.min(1, (v - centre) / span));
  const hue = t >= 0 ? 142 : 0;
  return `hsla(${hue}, 70%, 45%, ${0.12 + Math.abs(t) * 0.55})`;
}

export function Leaderboard({ onSelect }: { onSelect: (ticker: string) => void }) {
  const { data, error, loading } = useApi((s) => api.summary(s), "summary");
  const [metric, setMetric] = useState<Metric>("strat_sharpe");
  const m = METRICS[metric];

  const { stocks, cell, mean } = useMemo(() => {
    const rows = data ?? [];
    const stocks = [...new Set(rows.map((r) => r.ticker))].sort();
    const lookup = new Map(rows.map((r) => [`${r.ticker}|${r.model}`, r] as const));
    const cell = (t: string, model: string) => (lookup.get(`${t}|${model}`)?.[metric] as number | null | undefined) ?? null;
    const mean = (model: string) => {
      const vals = stocks.map((t) => cell(t, model)).filter((v): v is number => v != null);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    return { stocks, cell, mean };
  }, [data, metric]);

  if (error) return <ErrorBox message={error} />;
  if (loading || !data) return <div className="skeleton" style={{ height: 480 }} />;

  const buyHold = (t: string) => data.find((x: MetricRow) => x.ticker === t)?.bh_sharpe ?? null;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Leaderboard · all stocks, out-of-sample</h3>
        <div className="segmented">
          {(Object.keys(METRICS) as Metric[]).map((k) => (
            <button key={k} className={k === metric ? "active" : ""} onClick={() => setMetric(k)}>{METRICS[k].label}</button>
          ))}
        </div>
      </div>
      <div className="table-wrap">
        <table className="table heatmap">
          <thead>
            <tr>
              <th>Stock</th>
              {MODELS.map((model) => <th key={model}>{modelLabel(model)}</th>)}
              {metric === "strat_sharpe" && <th>Buy & hold</th>}
            </tr>
          </thead>
          <tbody>
            {stocks.map((t) => (
              <tr key={t}>
                <td><button className="link" onClick={() => onSelect(t)}>{t}</button></td>
                {MODELS.map((model) => {
                  const v = cell(t, model);
                  return (
                    <td key={model} className="mono" style={v == null ? undefined : { background: heat(v, m.centre, m.span) }}>
                      {v == null ? "–" : m.fmt(v)}
                    </td>
                  );
                })}
                {metric === "strat_sharpe" && <td className="mono muted">{buyHold(t) == null ? "–" : m.fmt(buyHold(t)!)}</td>}
              </tr>
            ))}
            <tr className="row-mean">
              <td>Mean</td>
              {MODELS.map((model) => {
                const v = mean(model);
                return <td key={model} className="mono">{v == null ? "–" : m.fmt(v)}</td>;
              })}
              {metric === "strat_sharpe" && <td />}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted small">
        1,512 test days per stock (Sep 2020 – Oct 2026). Green = better than {metric === "dir_acc" ? "a coin flip" :
        metric === "strat_sharpe" ? "zero" : "buy & hold"}. Click a stock to open it.
      </p>
    </div>
  );
}
