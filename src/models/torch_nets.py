"""PyTorch sequence models (LSTM, Transformer encoder) sharing one training loop.

Both see the last `seq_len` days of standardised features and predict the
standardised next-day log return. Scalers are fit on training rows only.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.models.base import Forecaster
from src.models.dataset import Dataset, make_windows, windows_ending_at

# Small sequence models gain little from intra-op threads; parallel jobs set TORCH_NUM_THREADS=1.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", os.cpu_count() or 1)))


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, units: int = 24, dropout: float = 0.35, dense_units: int = 8,
                 num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, units, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units, dense_units), nn.ReLU(),
                                  nn.Linear(dense_units, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerNet(nn.Module):
    """Linear projection -> sinusoidal positions -> pre-norm encoder -> last-step head."""

    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.1, dense_units: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, dense_units), nn.ReLU(),
                                  nn.Linear(dense_units, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.proj(x)))
        return self.head(self.norm(h[:, -1])).squeeze(-1)


TRAINING_KEYS = {"learning_rate", "epochs", "batch_size", "patience", "weight_decay"}


class TorchSequenceForecaster(Forecaster):
    net_class: type[nn.Module]

    def _net(self, n_features: int) -> nn.Module:
        arch = {k: v for k, v in self.params.items() if k not in TRAINING_KEYS}
        return self.net_class(n_features, **arch)

    def _windows(self, ds: Dataset) -> np.ndarray:
        return make_windows(self.feat_scaler_.transform(ds.X), self.seq_len)

    def fit(self, ds: Dataset, train: np.ndarray, val: np.ndarray) -> TorchSequenceForecaster:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        p = self.params

        # Scalers see only rows up to the end of the training block.
        self.feat_scaler_ = StandardScaler().fit(ds.X[: train[-1] + 1])
        self.tgt_scaler_ = StandardScaler().fit(ds.y[train].reshape(-1, 1))
        self.n_features_ = ds.X.shape[1]
        windows = self._windows(ds)

        def tensors(rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.tgt_scaler_.transform(ds.y[rows].reshape(-1, 1)).ravel()
            return (torch.from_numpy(windows_ending_at(windows, rows, self.seq_len)),
                    torch.from_numpy(y.astype("float32")))

        X_tr, y_tr = tensors(train)
        X_va, y_va = tensors(val)

        self.net_ = self._net(self.n_features_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=p.get("learning_rate", 7e-4),
                               weight_decay=p.get("weight_decay", 0.0))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, min_lr=1e-5)
        loss_fn = nn.HuberLoss(delta=1.0)
        batch, patience = p.get("batch_size", 32), p.get("patience", 15)

        best_loss, best_state, stale = math.inf, None, 0
        self.history_: list[tuple[float, float]] = []
        for _epoch in range(p.get("epochs", 120)):
            self.net_.train()
            order = rng.permutation(len(X_tr))
            total = 0.0
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net_(X_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            val_loss = self._loss(X_va, y_va, loss_fn)
            self.history_.append((total / len(X_tr), val_loss))
            sched.step(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        return self

    @torch.no_grad()
    def _loss(self, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> float:
        self.net_.eval()
        return float(loss_fn(self.net_(X), y))

    @torch.no_grad()
    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        self.net_.eval()
        X = torch.from_numpy(windows_ending_at(self._windows(ds), rows, self.seq_len))
        pred = self.net_(X).numpy().reshape(-1, 1)
        return self.tgt_scaler_.inverse_transform(pred).ravel().astype("float64")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net_.state_dict(), directory / "net.pt")
        joblib.dump({k: v for k, v in self.__dict__.items() if k != "net_"}, directory / "state.joblib")

    @classmethod
    def load(cls, directory: Path) -> TorchSequenceForecaster:
        obj = cls.__new__(cls)
        obj.__dict__.update(joblib.load(directory / "state.joblib"))
        obj.net_ = obj._net(obj.n_features_)
        obj.net_.load_state_dict(torch.load(directory / "net.pt", weights_only=True))
        obj.net_.eval()
        return obj


class TorchLSTM(TorchSequenceForecaster):
    name = "lstm"
    net_class = LSTMNet


class TorchTransformer(TorchSequenceForecaster):
    name = "transformer"
    net_class = TransformerNet


def describe(model: TorchSequenceForecaster) -> dict[str, Any]:
    return {"parameters": sum(p.numel() for p in model.net_.parameters()), "epochs": len(model.history_),
            "best_val_loss": model.best_val_loss_}
