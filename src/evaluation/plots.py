from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import strategy_returns  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def equity_curves(preds: pd.DataFrame, ticker: str, cfg: dict[str, Any], path: Path) -> None:
    """Net-of-cost long/flat equity per model vs buy-and-hold over the whole out-of-sample period."""
    st = cfg["strategy"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bh_drawn = False
    for model, g in preds.sort_values("date").groupby("model", sort=False):
        if model == "random_walk":  # never invests: a flat line at 1.0
            continue
        y, p = g["actual"].to_numpy(), g["pred"].to_numpy()
        daily, _ = strategy_returns(y, p, st["signal_threshold"], st["cost_per_side"])
        ax.plot(g["date"], np.cumprod(1 + daily), label=model, lw=1.2)
        if not bh_drawn:
            bh, _ = strategy_returns(y, np.ones_like(y), 0.0, st["cost_per_side"])
            ax.plot(g["date"], np.cumprod(1 + bh), label="buy & hold", color="black", lw=1.8)
            bh_drawn = True
    ax.set_title(f"{ticker} — long/flat equity, walk-forward out-of-sample, {st['cost_per_side']:.2%} cost/side")
    ax.set_ylabel("Growth of 1")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def accuracy_bars(metrics: pd.DataFrame, ticker: str, path: Path) -> None:
    """Out-of-sample directional accuracy per model and fold."""
    m = metrics[(metrics["ticker"] == ticker) & (metrics["fold"] != "all")].dropna(subset=["dir_acc"])
    pivot = m.pivot(index="fold", columns="model", values="dir_acc")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.axhline(50, color="black", lw=1, ls="--")
    ax.set_ylim(35, 65)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_xlabel("Walk-forward fold")
    ax.set_title(f"{ticker} — directional accuracy by fold")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summary_heatmaps(metrics: pd.DataFrame, path: Path) -> None:
    """Ticker x model heatmaps of out-of-sample directional accuracy and net Sharpe."""
    allf = metrics[metrics["fold"] == "all"]
    panels = [
        ("dir_acc", "Directional accuracy (%)", "RdYlGn", 50),
        ("strat_sharpe", "Long/flat Sharpe (net)", "RdYlGn", 0),
        ("excess_cagr_vs_bh", "CAGR minus buy & hold", "RdYlGn", 0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 0.45 * allf["ticker"].nunique() + 2))
    for ax, (col, title, cmap, centre) in zip(axes, panels, strict=True):
        pivot = allf.pivot(index="ticker", columns="model", values=col)
        span = np.nanmax(np.abs(pivot.to_numpy() - centre)) or 1.0
        im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=centre - span, vmax=centre + span, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=40, ha="right")
        ax.set_yticks(range(pivot.shape[0]), pivot.index)
        for (i, j), v in np.ndenumerate(pivot.to_numpy()):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
