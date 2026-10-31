from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("yfinance", "urllib3", "matplotlib", "h5py", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
