"""Expanding-window walk-forward splits with an embargo.

Row t of the feature frame is "today"; its label is the return from close t to
close t+1. A train row's label therefore touches row t+1, so blocks are
separated by `embargo` rows to keep any label from reaching into the next block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold: int
    train: np.ndarray  # row positions used to fit models/scalers
    val: np.ndarray    # early-stopping rows (after train, before test)
    test: np.ndarray   # out-of-sample rows


def walk_forward_folds(
    n_rows: int,
    n_folds: int,
    test_size: int,
    val_fraction: float = 0.15,
    embargo: int = 5,
    min_start: int = 0,
) -> list[Fold]:
    """Consecutive test blocks covering the last `n_folds * test_size` rows.

    Fold k trains on everything from `min_start` up to its test block (minus the
    validation tail and the embargo gaps), so the training window expands.
    """
    if n_folds < 1 or test_size < 1:
        raise ValueError("n_folds and test_size must be positive")

    first_test = n_rows - n_folds * test_size
    folds: list[Fold] = []
    for k in range(n_folds):
        test_start = first_test + k * test_size
        dev_end = test_start - embargo
        n_val = int((dev_end - min_start) * val_fraction)
        val_start = dev_end - n_val
        train_end = val_start - embargo
        if train_end - min_start < 2 * test_size or n_val < 1:
            raise ValueError(
                f"Not enough rows ({n_rows}) for {n_folds} folds of {test_size}: "
                f"fold {k} would train on {train_end - min_start} rows"
            )
        folds.append(
            Fold(
                fold=k,
                train=np.arange(min_start, train_end),
                val=np.arange(val_start, dev_end),
                test=np.arange(test_start, test_start + test_size),
            )
        )
    return folds
