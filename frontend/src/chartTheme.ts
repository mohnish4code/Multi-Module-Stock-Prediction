import { ColorType, CrosshairMode, type DeepPartial, type ChartOptions } from "lightweight-charts";

export interface Palette {
  bg: string;
  text: string;
  muted: string;
  grid: string;
  up: string;
  down: string;
  accent: string;
}

/** Reads the current CSS theme variables so charts follow the light/dark toggle. */
export function readPalette(): Palette {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    bg: v("--surface"),
    text: v("--text"),
    muted: v("--muted"),
    grid: v("--grid"),
    up: v("--up"),
    down: v("--down"),
    accent: v("--accent"),
  };
}

export function chartOptions(p: Palette): DeepPartial<ChartOptions> {
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: p.bg },
      textColor: p.muted,
      fontFamily: "Inter, system-ui, sans-serif",
      attributionLogo: false,
    },
    grid: { vertLines: { color: p.grid }, horzLines: { color: p.grid } },
    rightPriceScale: { borderColor: p.grid },
    timeScale: { borderColor: p.grid, rightOffset: 4 },
    crosshair: { mode: CrosshairMode.Normal },
  };
}
