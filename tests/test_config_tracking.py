from __future__ import annotations

from pathlib import Path

import pytest

from src import config, tracking


def test_tuned_params_override_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text("models:\n  xgboost:\n    max_depth: 5\n  newmodel:\n    a: 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "TUNED_CONFIG", tuned)
    cfg = config.load_config()
    assert cfg["models"]["xgboost"]["max_depth"] == 5
    assert cfg["models"]["xgboost"]["subsample"] == 0.8  # untouched keys survive
    assert cfg["models"]["newmodel"] == {"a": 1}
    assert config.load_config(use_tuned=False)["models"]["xgboost"]["max_depth"] == 3


def test_tracking_is_a_no_op_when_disabled() -> None:
    with tracking.run("x", "y") as r:
        r.params({"a": {"b": 1}})
        r.metrics({"m": 1.0, "nan": float("nan"), "s": "text"})
        r.artifact(Path("does-not-exist"))


def test_flatten() -> None:
    assert tracking._flatten({"a": {"b": 1, "c": [1, 2]}, "d": None}) == {"a.b": 1, "a.c": "[1, 2]", "d": None}
