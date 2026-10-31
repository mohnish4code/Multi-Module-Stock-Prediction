import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";

import { chartOptions, readPalette } from "../chartTheme";
import { nextBusinessDay, price } from "../format";
import type { Bar, Forecast } from "../types";

const RANGES = { "1M": 22, "3M": 66, "6M": 130, "1Y": 260 } as const;
type Range = keyof typeof RANGES;
type Overlay = "sma" | "ema" | "bands" | "volume";

const OVERLAY_LABELS: Record<Overlay, string> = { sma: "SMA 20", ema: "EMA 20", bands: "Bollinger", volume: "Volume" };

export function PriceChart({ history, forecast, theme }: { history: Bar[]; forecast: Forecast; theme: string }) {
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [range, setRange] = useState<Range>("3M");
  const [overlays, setOverlays] = useState<Set<Overlay>>(new Set(["sma", "ema", "bands", "volume"]));

  const bars = useMemo(() => history.filter((b) => b.close != null && b.open != null), [history]);

  useEffect(() => {
    if (!container.current) return;
    const p = readPalette();
    const chart = createChart(container.current, chartOptions(p));
    chartRef.current = chart;

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: p.up, downColor: p.down, wickUpColor: p.up, wickDownColor: p.down, borderVisible: false,
    });
    candles.setData(bars.map((b) => ({ time: b.date, open: b.open!, high: b.high!, low: b.low!, close: b.close! })));

    const line = (key: keyof Bar, color: string, width = 1) => {
      const s = chart.addSeries(LineSeries, { color, lineWidth: width as 1, priceLineVisible: false, lastValueVisible: false });
      s.setData(bars.filter((b) => b[key] != null).map((b) => ({ time: b.date, value: b[key] as number })));
    };
    if (overlays.has("sma")) line("sma_20", "#f0b90b", 2);
    if (overlays.has("ema")) line("ema_20", "#a78bfa", 2);
    if (overlays.has("bands")) {
      line("bb_upper", "rgba(96,165,250,0.55)");
      line("bb_lower", "rgba(96,165,250,0.55)");
    }
    if (overlays.has("volume")) {
      const vol = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "" });
      vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      vol.setData(bars.map((b) => ({
        time: b.date, value: b.volume ?? 0, color: (b.close! >= b.open! ? p.up : p.down) + "55",
      })));
    }

    // Forecast: dashed segment from the last close to the predicted next close.
    const last = bars[bars.length - 1];
    if (last) {
      const next = nextBusinessDay(last.date);
      const fc = chart.addSeries(LineSeries, {
        color: p.accent, lineWidth: 2, lineStyle: LineStyle.Dashed, priceLineVisible: false, title: "forecast",
      });
      fc.setData([{ time: last.date, value: last.close! }, { time: next, value: forecast.predicted_price }]);
      createSeriesMarkers(fc, [{
        time: next, position: "inBar", color: p.accent, shape: "circle", text: price(forecast.predicted_price),
      }]);
    }

    const n = RANGES[range];
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, bars.length - n), to: bars.length + 3 });
    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, forecast, overlays, range, theme]);

  const toggle = (o: Overlay) =>
    setOverlays((prev) => {
      const next = new Set(prev);
      if (next.has(o)) next.delete(o);
      else next.add(o);
      return next;
    });

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Price & forecast</h3>
        <div className="controls">
          <div className="segmented">
            {(Object.keys(RANGES) as Range[]).map((r) => (
              <button key={r} className={r === range ? "active" : ""} onClick={() => setRange(r)}>{r}</button>
            ))}
          </div>
          <div className="chips">
            {(Object.keys(OVERLAY_LABELS) as Overlay[]).map((o) => (
              <button key={o} className={`chip ${overlays.has(o) ? "on" : ""}`} onClick={() => toggle(o)}>
                {OVERLAY_LABELS[o]}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div ref={container} className="chart chart-tall" />
    </div>
  );
}
