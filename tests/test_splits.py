from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.evaluation.splits import walk_forward_folds
from src.models.torch_nets import TorchLSTM

N, FOLDS, TEST, EMBARGO, START = 3000, 6, 252, 5, 59


@pytest.fixture
def folds():
    return walk_forward_folds(N, FOLDS, TEST, val_fraction=0.15, embargo=EMBARGO, min_start=START)


def test_fold_count_and_test_blocks_tile_the_end(folds) -> None:
    assert len(folds) == FOLDS >= 5
    tests = np.concatenate([f.test for f in folds])
    assert np.array_equal(tests, np.arange(N - FOLDS * TEST, N))


def test_chronological_with_embargo(folds) -> None:
    for f in folds:
        assert f.train[0] == START
        # a row's label uses the next row, so the gap must be > 1 row
        assert f.val[0] - f.train[-1] == EMBARGO + 1
        assert f.test[0] - f.val[-1] == EMBARGO + 1
        assert np.all(np.diff(f.train) == 1)
        assert np.all(np.diff(f.val) == 1)
        assert not set(f.train) & set(f.val)
        assert not set(f.val) & set(f.test)


def test_training_window_expands(folds) -> None:
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_too_little_data_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_folds(1000, 6, 252)


def test_scaler_sees_only_training_rows(dataset, monkeypatch) -> None:
    """The LSTM's feature scaler is fit on rows up to the end of the training block only."""
    fits = []
    original = StandardScaler.fit

    def spy(self, X, y=None):
        fits.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    model = TorchLSTM(seq_len=10)
    train, val = np.arange(9, 400), np.arange(410, 480)

    def stop(*_args, **_kwargs):
        raise StopIteration  # don't train; only the scaler fitting matters here

    monkeypatch.setattr(TorchLSTM, "_net", stop)
    with pytest.raises(StopIteration):
        model.fit(dataset, train, val)
    assert fits == [train[-1] + 1, len(train)]
