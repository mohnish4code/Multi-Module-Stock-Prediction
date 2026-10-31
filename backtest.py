"""Walk-forward backtest of every configured model on every configured ticker.

    python backtest.py                         # all tickers, all models
    python backtest.py --tickers TCS INFY      # subset
    python backtest.py --models xgboost lstm   # subset of models
    python backtest.py --features news         # add FinBERT news features (models labelled <model>+news)
    python backtest.py --refresh               # re-download the data snapshot
    python backtest.py --replot                # rescore saved predictions and redraw plots only

Every run is logged to MLflow (experiment "backtest"): one nested run per
ticker x model with its parameters, per-fold and overall metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path, resolve_ticker
from src.evaluation import plots
from src.evaluation.backtest import (
    load_dataset,
    merge_into_csv,
    merge_predictions,
    predict_walk_forward,
    score_predictions,
)
from src.logging_setup import setup_logging
from src.tracking import run

logger = logging.getLogger("backtest")

LOGGED = ["dir_acc", "p_vs_50", "rmse", "mae", "mape", "strat_sharpe", "strat_sharpe_gross", "strat_cagr",
          "strat_max_dd", "bh_sharpe", "bh_cagr", "excess_cagr_vs_bh", "exposure", "n_trades"]


def log_scores(cfg: dict, folder: str, scores: pd.DataFrame, models: list[str], suffix: str, features: str) -> None:
    for name in models:
        label = name + suffix
        rows = scores[scores["model"] == label]
        with run("backtest", f"{folder}/{label}", nested=True) as r:
            r.tags({"ticker": folder, "model": name, "feature_set": features})
            r.params({"model": cfg["models"].get(name, {}), "backtest": cfg["backtest"], "strategy": cfg["strategy"]})
            for _, row in rows[rows["fold"] != "all"].iterrows():
                r.metrics({k: row[k] for k in ("dir_acc", "strat_sharpe", "mape")}, step=int(row["fold"]))
            overall = rows[rows["fold"] == "all"]
            if not overall.empty:
                r.metrics({k: overall.iloc[0][k] for k in LOGGED})


def rescore_and_plot(cfg: dict, tickers: list[str], results: Path) -> pd.DataFrame | None:
    """Recomputes metrics.csv and every plot from the saved predictions, without refitting anything."""
    all_metrics = None
    for ticker in tickers:
        folder = cfg["tickers"][ticker]["folder"]
        path = results / "predictions" / f"{folder}.parquet"
        if not path.exists():
            logger.warning("No predictions for %s", folder)
            continue
        preds = pd.read_parquet(path)
        all_metrics = merge_into_csv(score_predictions(preds, folder, cfg), results / "metrics.csv")
        plots.equity_curves(preds, folder, cfg, results / "plots" / f"{folder}_equity.png")
        plots.accuracy_bars(all_metrics, folder, results / "plots" / f"{folder}_direction.png")
    if all_metrics is not None:
        plots.summary_heatmaps(all_metrics, results / "plots" / "summary_heatmap.png")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", help="default: all tickers in the config")
    parser.add_argument("--models", nargs="*", help="default: backtest.models in the config")
    parser.add_argument("--features", choices=["base", "news"], default="base")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="re-download the OHLCV snapshot")
    parser.add_argument("--replot", action="store_true", help="only rescore saved predictions and redraw plots")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    tickers = [resolve_ticker(cfg, t) for t in args.tickers] if args.tickers else list(cfg["tickers"])
    models = args.models or cfg["backtest"]["models"]
    suffix = "" if args.features == "base" else f"+{args.features}"
    results = resolve_path(cfg["output"]["results_dir"])

    if args.replot:
        rescore_and_plot(cfg, tickers, results)
        return

    all_metrics = None
    with run("backtest", f"backtest {args.features} {'/'.join(models)}") as parent:
        parent.params({"tickers": ",".join(tickers), "models": ",".join(models), "feature_set": args.features})
        for ticker in tickers:
            folder = cfg["tickers"][ticker]["folder"]
            ds = load_dataset(cfg, ticker, refresh=args.refresh, feature_set=args.features)
            logger.info("%s: %d rows %s..%s, %d features", ticker, len(ds.y), ds.dates[0].date(),
                        ds.dates[-1].date(), len(ds.feature_names))

            new = predict_walk_forward(cfg, ds, models, suffix)
            preds = merge_predictions(new, results / "predictions" / f"{folder}.parquet")
            scores = score_predictions(preds, folder, cfg)
            all_metrics = merge_into_csv(scores, results / "metrics.csv")
            log_scores(cfg, folder, scores, models, suffix, args.features)

            plots.equity_curves(preds, folder, cfg, results / "plots" / f"{folder}_equity.png")
            plots.accuracy_bars(all_metrics, folder, results / "plots" / f"{folder}_direction.png")

        if all_metrics is not None:
            plots.summary_heatmaps(all_metrics, results / "plots" / "summary_heatmap.png")
            parent.artifact(results / "metrics.csv")
            parent.artifact(results / "plots", "plots")
            summary = all_metrics[all_metrics["fold"] == "all"]
            cols = ["dir_acc", "p_vs_50", "mape", "strat_sharpe", "strat_cagr", "bh_cagr", "excess_cagr_vs_bh"]
            with pd.option_context("display.width", 160, "display.max_rows", 200):
                logger.info("\n%s", summary.groupby("model")[cols].mean().round(4))


if __name__ == "__main__":
    main()
