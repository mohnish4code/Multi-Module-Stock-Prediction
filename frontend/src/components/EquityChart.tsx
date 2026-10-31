import { LineSeries, createChart } from "lightweight-charts";
import { useEffect, useRef } from "react";

import { chartOptions, readPalette } from "../chartTheme";
import { MODEL_COLORS, modelLabel } from "../format";

interface Props {
  dates: string[];
  curves: Record<string, number[]>;
  visible: Set<string>;
  onToggle: (model: string) => void;
  theme: string;
}

export function EquityChart({ dates, curves, visible, onToggle, theme }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const models = Object.keys(curves).filter((m) => m !== "random_walk");

  useEffect(() => {
    if (!container.current) return;
    const chart = createChart(container.current, {
      ...chartOptions(readPalette()),
      localization: { priceFormatter: (v: number) => `${v.toFixed(2)}×` },
    });
    for (const model of models) {
      if (!visible.has(model)) continue;
      const series = chart.addSeries(LineSeries, {
        color: MODEL_COLORS[model] ?? "#94a3b8",
        lineWidth: model === "buy_and_hold" ? 3 : 2,
        priceLineVisible: false,
        title: modelLabel(model),
        lastValueVisible: true,
      });
      series.setData(dates.map((d, i) => ({ time: d, value: curves[model][i] })));
    }
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [dates, curves, visible, theme]);

  return (
    <>
      <div className="legend">
        {models.map((m) => {
          const final = curves[m][curves[m].length - 1];
          return (
            <button
              key={m}
              className={`legend-item ${visible.has(m) ? "on" : ""}`}
              onClick={() => onToggle(m)}
              title="Show / hide"
            >
              <span className="swatch" style={{ background: MODEL_COLORS[m] ?? "#94a3b8" }} />
              {modelLabel(m)}
              <span className={`mono ${final >= 1 ? "up" : "down"}`}>{final.toFixed(2)}×</span>
            </button>
          );
        })}
      </div>
      <div ref={container} className="chart" />
    </>
  );
}
