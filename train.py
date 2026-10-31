"""Trains models on all history and saves them to models/<TICKER>/<label>/ (e.g. xgboost+news).

    python train.py                          # the serving model for every ticker (config: serving)
    python train.py --tickers TCS INFY       # subset
    python train.py --models lstm --features base
    python train.py --refresh                # re-download the price snapshot first

The final model uses all history except a validation tail (for early stopping),
separated by the same embargo as the backtest. Each model directory gets a
manifest.json with the data range, features, parameters, git commit and the
model's walk-forward backtest metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_path, resolve_ticker
from src.evaluation.backtest import load_dataset
from src.logging_setup import setup_logging
from src.models.dataset import Dataset
from src.models.registry import create_model, model_dir, variant
from src.tracking import run

logger = logging.getLogger("train")


def final_split(cfg: dict[str, Any], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    start = cfg["features"]["sequence_length"] - 1
    embargo = cfg["backtest"]["embargo"]
    n_val = int((n_rows - start) * cfg["backtest"]["val_fraction"])
    val_start = n_rows - n_val
    return np.arange(start, val_start - embargo), np.arange(val_start, n_rows)


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,  # noqa: S607
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def backtest_metrics(cfg: dict[str, Any], folder: str, model: str) -> dict[str, Any] | None:
    path = resolve_path(cfg["output"]["results_dir"]) / "metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, dtype={"fold": str})
    row = m[(m["ticker"] == folder) & (m["model"] == model) & (m["fold"] == "all")]
    return None if row.empty else row.iloc[0].replace({np.nan: None}).to_dict()


def train_one(cfg: dict[str, Any], ds: Dataset, folder: str, name: str, feature_set: str) -> None:
    label = variant(name, feature_set)
    train, val = final_split(cfg, len(ds.y))
    started = time.perf_counter()
    model = create_model(cfg, name).fit(ds, train, val)
    out = model_dir(resolve_path(cfg["serving"]["models_dir"]), folder, label)
    model.save(out)
    manifest = {
        "ticker": ds.ticker,
        "model": label,
        "feature_set": feature_set,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "data_start": ds.dates[0].date().isoformat(),
        "data_end": ds.dates[-1].date().isoformat(),
        "train_rows": len(train),
        "val_rows": len(val),
        "features": ds.feature_names,
        "sequence_length": model.seq_len,
        "params": model.params,
        "target": "next_day_log_return",
        "backtest": backtest_metrics(cfg, folder, label),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("%s %-12s trained in %.1fs -> %s", ds.ticker, label, time.perf_counter() - started, out)

    with run("training", f"{folder}/{label}") as r:
        r.tags({"ticker": folder, "model": label, "git_sha": manifest["git_sha"] or "unknown"})
        r.params({"params": model.params, "train_rows": len(train), "val_rows": len(val),
                  "data_end": manifest["data_end"], "n_features": len(ds.feature_names)})
        r.metrics({f"backtest_{k}": v for k, v in (manifest["backtest"] or {}).items()
                   if k in ("dir_acc", "strat_sharpe", "mape", "excess_cagr_vs_bh")})
        r.artifact(out / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--models", nargs="*", help="default: serving.model")
    parser.add_argument("--features", choices=["base", "news"], help="default: serving.feature_set")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    tickers = [resolve_ticker(cfg, t) for t in args.tickers] if args.tickers else list(cfg["tickers"])
    models = args.models or [cfg["serving"]["model"]]
    feature_set = args.features or cfg["serving"]["feature_set"]

    failures = []
    for ticker in tickers:
        ds = load_dataset(cfg, ticker, refresh=args.refresh, feature_set=feature_set)
        for name in models:
            try:
                train_one(cfg, ds, cfg["tickers"][ticker]["folder"], name, feature_set)
            except Exception:  # keep going; report at the end
                logger.exception("Training %s %s failed", ticker, name)
                failures.append(f"{ticker}/{name}")
    if failures:
        raise SystemExit(f"Failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
