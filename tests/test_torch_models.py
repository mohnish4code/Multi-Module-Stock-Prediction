from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.dataset import Dataset
from src.models.torch_nets import TorchLSTM, TorchTransformer

TRAIN, VAL, TEST = np.arange(19, 500), np.arange(506, 580), np.arange(586, 650)
FAST = {"epochs": 2, "batch_size": 64, "patience": 2}


@pytest.mark.parametrize(("cls", "arch"), [
    (TorchLSTM, {"units": 8, "dense_units": 4}),
    (TorchTransformer, {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 16}),
])
def test_fit_predict_save_load(cls, arch, dataset: Dataset, tmp_path: Path) -> None:
    model = cls({**arch, **FAST}, seq_len=20, seed=0).fit(dataset, TRAIN, VAL)
    pred = model.predict(dataset, TEST)
    assert pred.shape == (len(TEST),)
    assert np.isfinite(pred).all()
    assert 1 <= len(model.history_) <= 2

    model.save(tmp_path)
    np.testing.assert_allclose(cls.load(tmp_path).predict(dataset, TEST), pred, rtol=1e-5, atol=1e-9)


def test_same_seed_same_predictions(dataset: Dataset) -> None:
    def fit() -> np.ndarray:
        return TorchLSTM({"units": 8, **FAST}, seq_len=20, seed=7).fit(dataset, TRAIN, VAL).predict(dataset, TEST)

    np.testing.assert_allclose(fit(), fit())


def test_prediction_uses_only_past_rows(dataset: Dataset) -> None:
    model = TorchTransformer({"d_model": 16, "nhead": 2, "num_layers": 1, **FAST}, seq_len=20).fit(dataset, TRAIN, VAL)
    before = model.predict(dataset, TEST)
    tampered = Dataset(dataset.ticker, dataset.dates, dataset.X.copy(), dataset.y, dataset.close,
                       dataset.feature_names)
    tampered.X[TEST[-1] + 1 :] = 99.0
    np.testing.assert_allclose(model.predict(tampered, TEST), before)
