import { useMemo, useState } from "react";

import { BASELINES, MODEL_COLORS, modelLabel, num, pct } from "../format";
import type { Backtest, MetricRow } from "../types";
import { EquityChart } from "./EquityChart";

const DEFAULT_VISIBLE = ["buy_and_hold", "xgboost+news", "xgboost", "lstm", "transformer", "arima"];

type Col = { key: keyof MetricRow; label: string; fmt: (v: number | null) => string; better: "high" | "low" };
const COLUMNS: Col[] = [
  { key: "mape", label: "MAPE", fmt: (v) => (v == null ? "–" : `${v.toFixed(3)}%`), better: "low" },
  { key: "strat_sharpe", label: "Sharpe (net)", fmt: (v) => num(v), better: "high" },
  { key: "strat_cagr", label: "CAGR", fmt: (v) => pct(v), better: "high" },
  { key: "strat_max_dd", label: "Max drawdown", fmt: (v) => pct(v), better: "high" },
  { key: "exposure", label: "Time in market", fmt: (v) => pct(v, 0), better: "high" },
];

export function BacktestView({ data, theme }: { data: Backtest; theme: string }) {
  const [visible, setVisible] = useState(() => new Set(DEFAULT_VISIBLE.filter((m) => m in data.equity.curves)));
  const toggle = (m: string) =>
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(m)) next.delete(m);
      else next.add(m);
      return next;
    });
  const { dates } = data.equity;

  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-head">
          <h3>Walk-forward equity · long/flat, net of costs</h3>
          <span className="muted small">
            {dates[0]} → {dates[dates.length - 1]} · retrained every year · {(data.cost_per_side * 100).toFixed(2)}% per side
          </span>
        </div>
        <EquityChart dates={dates} curves={data.equity.curves} visible={visible} onToggle={toggle} theme={theme} />
      </div>
      <MetricsTable rows={data.metrics} />
      <YearlyReturns rows={data.metrics} visible={visible} />
    </div>
  );
}

function MetricsTable({ rows }: { rows: MetricRow[] }) {
  const [sort, setSort] = useState<{ key: keyof MetricRow; desc: boolean }>({ key: "strat_sharpe", desc: true });
  const all = rows.filter((r) => r.fold === "all");
  const bh = all[0];

  const table = useMemo(() => {
    const models = all.map((r) => ({ ...r }));
    if (bh) {
      models.push({
        ...bh, model: "buy_and_hold", dir_acc: null, p_vs_50: null, mape: null, rmse: null,
        strat_sharpe: bh.bh_sharpe, strat_cagr: bh.bh_cagr, strat_max_dd: bh.bh_max_dd ?? null, exposure: 1,
      });
    }
    return models.sort((a, b) => {
      const av = a[sort.key] as number | null;
      const bv = b[sort.key] as number | null;
      if (av == null) return 1;
      if (bv == null) return -1;
      return sort.desc ? bv - av : av - bv;
    });
  }, [all, bh, sort]);

  const best = (col: Col) => {
    const values = table.filter((r) => r.model !== "buy_and_hold").map((r) => r[col.key] as number | null)
      .filter((v): v is number => v != null);
    return col.better === "high" ? Math.max(...values) : Math.min(...values);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Models vs baselines</h3>
        <span className="muted small">Click a column to sort · best model value highlighted</span>
      </div>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="sortable" onClick={() =>
                  setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : true }))}>
                  {c.label} {sort.key === c.key ? (sort.desc ? "↓" : "↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.map((r) => (
              <tr key={r.model} className={r.model === "buy_and_hold" ? "row-bh" : ""}>
                <td>
                  <span className="swatch" style={{ background: MODEL_COLORS[r.model] ?? "#94a3b8" }} />
                  {modelLabel(r.model)}
                  {BASELINES.has(r.model) && <span className="tag">baseline</span>}
                </td>
                {COLUMNS.map((c) => {
                  const v = r[c.key] as number | null;
                  const isBest = v != null && r.model !== "buy_and_hold" && v === best(c);
                  return <td key={c.key} className={`mono ${isBest ? "best" : ""}`}>{c.fmt(v)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function YearlyReturns({ rows, visible }: { rows: MetricRow[]; visible: Set<string> }) {
  const folds = [...new Set(rows.filter((r) => r.fold !== "all").map((r) => r.fold))].sort();
  const series = [...visible].filter((m) => m === "buy_and_hold" || rows.some((r) => r.model === m));
  const value = (model: string, fold: string) => {
    const row = rows.find((r) => r.fold === fold && (model === "buy_and_hold" || r.model === model));
    if (!row) return null;
    return (model === "buy_and_hold" ? row.bh_total_return : row.strat_total_return) ?? null;
  };
  const year = (fold: string) => rows.find((r) => r.fold === fold)?.test_start.slice(0, 4) ?? fold;

  const values = folds.flatMap((f) => series.map((m) => value(m, f))).filter((v): v is number => v != null);
  const top = Math.max(0.1, ...values);
  const bottom = Math.min(-0.1, ...values);
  const W = 760, H = 240, pad = 40;
  const groupW = (W - pad * 2) / Math.max(1, folds.length);
  const barW = Math.max(4, (groupW - 16) / Math.max(1, series.length));
  const y = (v: number) => pad / 2 + ((top - v) / (top - bottom)) * (H - pad * 1.5);
  const ticks = [bottom, bottom / 2, 0, top / 2, top];

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Return by test year</h3>
        <span className="muted small">Strategy return per year vs buy & hold · models follow the equity legend</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="folds" role="img" aria-label="Return per test year">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad} x2={W - pad / 2} y1={y(t)} y2={y(t)} className={t === 0 ? "axis-50" : "axis"} />
            <text x={2} y={y(t) + 4} className="axis-label">{`${(t * 100).toFixed(0)}%`}</text>
          </g>
        ))}
        {folds.map((f, i) => (
          <g key={f}>
            {series.map((m, j) => {
              const v = value(m, f);
              if (v == null) return null;
              return (
                <rect key={m} x={pad + i * groupW + 8 + j * barW} width={barW - 2} y={Math.min(y(v), y(0))}
                  height={Math.abs(y(v) - y(0))} fill={MODEL_COLORS[m] ?? "#94a3b8"} rx={2}>
                  <title>{`${modelLabel(m)} · ${year(f)}: ${(v * 100).toFixed(1)}%`}</title>
                </rect>
              );
            })}
            <text x={pad + i * groupW + groupW / 2} y={H - 6} textAnchor="middle" className="axis-label">{year(f)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
