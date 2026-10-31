from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: Path, timeout: float = 300.0) -> Iterator[None]:
    """Cross-process lock (a lock file created with O_EXCL) for files shared by parallel jobs."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Could not lock {path}") from None
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
