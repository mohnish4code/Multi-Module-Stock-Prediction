"""Model name -> Forecaster class. Heavy frameworks are imported only when asked for."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.base import Forecaster

BASELINES = ("random_walk", "persistence", "majority", "arima")


def get_model_class(name: str) -> type[Forecaster]:
    if name in BASELINES:
        from src.models import baselines

        return {c.name: c for c in (baselines.RandomWalk, baselines.Persistence,
                                    baselines.Majority, baselines.Arima)}[name]
    if name == "xgboost":
        from src.models.xgb import XGBoost

        return XGBoost
    if name in ("lstm", "transformer"):
        from src.models.torch_nets import TorchLSTM, TorchTransformer

        return TorchLSTM if name == "lstm" else TorchTransformer
    raise KeyError(f"Unknown model '{name}'")


def create_model(cfg: dict[str, Any], name: str, seed: int | None = None) -> Forecaster:
    cls = get_model_class(name)
    return cls(
        params=dict(cfg.get("models", {}).get(name, {})),
        seq_len=cfg["features"]["sequence_length"],
        seed=cfg["backtest"]["seed"] if seed is None else seed,
    )


def variant(name: str, feature_set: str = "base") -> str:
    """Label used in results and model folders: 'xgboost' or 'xgboost+news'."""
    return name if feature_set == "base" else f"{name}+{feature_set}"


def model_dir(models_root: Path, folder: str, label: str) -> Path:
    return models_root / folder / label


def load_model(models_root: Path, folder: str, label: str) -> tuple[Forecaster, dict[str, Any]]:
    """Loads a model saved by train.py together with its manifest."""
    directory = model_dir(models_root, folder, label)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No trained '{label}' model for {folder}. Run: python train.py --tickers {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return get_model_class(label.split("+")[0]).load(directory), manifest
