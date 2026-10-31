"""Loads configs/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
TUNED_CONFIG = PROJECT_ROOT / "configs" / "tuned.yaml"


def load_config(path: str | Path | None = None, use_tuned: bool = True) -> dict[str, Any]:
    """Loads the YAML config. Hyperparameters found by tune.py (configs/tuned.yaml) override
    the hand-set ones under `models` unless use_tuned=False."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if use_tuned and TUNED_CONFIG.exists():
        tuned = yaml.safe_load(TUNED_CONFIG.read_text(encoding="utf-8")) or {}
        for name, params in tuned.get("models", {}).items():
            cfg["models"].setdefault(name, {}).update(params)
    return cfg


def resolve_path(relative: str) -> Path:
    """Config paths are relative to the project root."""
    return PROJECT_ROOT / relative


def resolve_ticker(cfg: dict[str, Any], value: str) -> str:
    """Accepts 'TCS', 'tcs', 'TCS.NS' or the model folder name; returns the canonical ticker."""
    key = value.upper().strip()
    for ticker, info in cfg["tickers"].items():
        if key in (ticker, ticker.removesuffix(".NS"), info["folder"].upper()):
            return ticker
    raise KeyError(f"Unknown ticker '{value}'. Known: {', '.join(cfg['tickers'])}")
