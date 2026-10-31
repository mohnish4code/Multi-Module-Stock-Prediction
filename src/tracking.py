"""MLflow tracking with a no-op fallback.

Tracking URI comes from MLFLOW_TRACKING_URI (default: a local SQLite store,
sqlite:///mlflow.db). Set MLFLOW_DISABLE=1 to turn tracking off (tests, CI).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _mlflow():
    if os.getenv("MLFLOW_DISABLE", "").lower() in ("1", "true", "yes"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"))
    return mlflow


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v if isinstance(v, int | float | str | bool) or v is None else str(v)
    return out


class Run:
    """Logging calls never raise: a tracking hiccup (e.g. a locked SQLite store) must not kill a job."""

    def __init__(self, mlflow: Any) -> None:
        self._m = mlflow

    def _safe(self, fn: str, *args: Any, **kwargs: Any) -> None:
        if not self._m:
            return
        try:
            getattr(self._m, fn)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow %s failed: %s", fn, exc)

    def params(self, params: dict[str, Any]) -> None:
        self._safe("log_params", {k: str(v)[:500] for k, v in _flatten(params).items()})

    def metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)}
        self._safe("log_metrics", clean, step=step)

    def artifact(self, path: Path, folder: str | None = None) -> None:
        if path.exists():
            self._safe("log_artifacts" if path.is_dir() else "log_artifact", str(path), folder)

    def tags(self, tags: dict[str, str]) -> None:
        self._safe("set_tags", tags)


@contextmanager
def run(experiment: str, name: str, nested: bool = False) -> Iterator[Run]:
    mlflow = _mlflow()
    active = None
    if mlflow is not None:
        try:
            mlflow.set_experiment(experiment)
            active = mlflow.start_run(run_name=name, nested=nested)
        except Exception as exc:  # noqa: BLE001 - never fail a job because tracking is unavailable
            logger.warning("MLflow unavailable (%s); continuing without tracking", exc)
    try:
        yield Run(mlflow if active else None)
    finally:
        if active:
            try:
                mlflow.end_run()
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow end_run failed: %s", exc)
