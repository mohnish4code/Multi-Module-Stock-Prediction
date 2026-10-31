import { useMemo, useState } from "react";

import { BASELINES, MODEL_COLORS, modelLabel, num, pct } from "../format";
import type { Backtest, MetricRow } from "../types";
import { EquityChart } from "./EquityChart";

const DEFAULT_VISIBLE = ["buy_and_hold", "xgboost+news", "xgboost", "lstm", "transformer", "arima"];

type Col = { key: keyof MetricRow; label: string; fmt: (v: number | null) => string; better: "high" | "low" };
const COLUMNS: Col[] = [
  { key: "dir_acc", label: "Direction acc.", fmt: (v) => (v == null ? "–" : `${v.toFixed(1)}%`), better: "high" },
  { key: "p_vs_50", label: "p vs 50%", fmt: (v) => num(v, 3), better: "low" },
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
      <FoldBars rows={data.metrics} visible={visible} />
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

function FoldBars({ rows, visible }: { rows: MetricRow[]; visible: Set<string> }) {
  const folds = [...new Set(rows.filter((r) => r.fold !== "all").map((r) => r.fold))].sort();
  const models = [...visible].filter((m) => rows.some((r) => r.model === m && r.dir_acc != null));
  const value = (model: string, fold: string) => rows.find((r) => r.model === model && r.fold === fold)?.dir_acc ?? null;
  const start = (fold: string) => rows.find((r) => r.fold === fold)?.test_start.slice(0, 4) ?? fold;

  const W = 760, H = 220, pad = 30, lo = 40, hi = 60;
  const groupW = (W - pad * 2) / Math.max(1, folds.length);
  const barW = Math.max(4, (groupW - 16) / Math.max(1, models.length));
  const y = (v: number) => H - pad - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (H - pad * 2);

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Direction accuracy by test year</h3>
        <span className="muted small">Models shown follow the equity legend · dashed line = coin flip</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="folds" role="img" aria-label="Direction accuracy per fold">
        {[40, 45, 50, 55, 60].map((t) => (
          <g key={t}>
            <line x1={pad} x2={W - pad} y1={y(t)} y2={y(t)} className={t === 50 ? "axis-50" : "axis"} />
            <text x={4} y={y(t) + 4} className="axis-label">{t}%</text>
          </g>
        ))}
        {folds.map((f, i) => (
          <g key={f}>
            {models.map((m, j) => {
              const v = value(m, f);
              if (v == null) return null;
              const x = pad + i * groupW + 8 + j * barW;
              return (
                <rect key={m} x={x} width={barW - 2} y={Math.min(y(v), y(50))} height={Math.abs(y(v) - y(50))}
                  fill={MODEL_COLORS[m] ?? "#94a3b8"} rx={2}>
                  <title>{`${modelLabel(m)} · ${start(f)}: ${v.toFixed(1)}%`}</title>
                </rect>
              );
            })}
            <text x={pad + i * groupW + groupW / 2} y={H - 8} textAnchor="middle" className="axis-label">{start(f)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
