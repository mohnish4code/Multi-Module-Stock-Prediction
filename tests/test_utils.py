from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import file_lock


def test_file_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    target = tmp_path / "metrics.csv"
    with file_lock(target):
        assert (tmp_path / "metrics.csv.lock").exists()
        with pytest.raises(TimeoutError), file_lock(target, timeout=0.3):
            pass
    assert not (tmp_path / "metrics.csv.lock").exists()
    with file_lock(target, timeout=0.3):  # can be taken again after release
        pass
