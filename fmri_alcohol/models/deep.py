"""Deep-learning models on the raw (z-scored) BOLD time series, in PyTorch.

Architectures follow the reference study (window 5, stride 1, 5 conv filters,
RNN output dimension 5):

* ``rnn``          LSTM only, single output
* ``rnn_nn``       LSTM -> dense layer
* ``cnn_nn``       1-D CNN -> flatten -> dense layer
* ``cnn_rnn``      1-D CNN -> LSTM -> dense layer  (paper's main model)
* ``cnn_rnn_demo`` cnn_rnn + demographics concatenated before the dense layer
                   (the paper's proposed future work)

Training uses batch binary cross-entropy and the paper's three dev-set
stopping rules (see ``should_stop``).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

ARCHITECTURES = ("rnn", "rnn_nn", "cnn_nn", "cnn_rnn", "cnn_rnn_demo")


class SeqNet(nn.Module):
    def __init__(self, arch: str, n_regions: int, n_timepoints: int, n_demo: int = 0,
                 filters: int = 5, window: int = 5, rnn_dim: int = 5, dropout: float = 0.5):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch}")
        self.arch = arch
        self.use_cnn = arch.startswith("cnn")
        self.use_rnn = "rnn" in arch
        self.use_demo = arch.endswith("demo")
        self.drop = nn.Dropout(dropout)
        in_dim = n_regions
        if self.use_cnn:
            self.conv = nn.Conv1d(n_regions, filters, kernel_size=window, stride=1)
            in_dim = filters
        if self.use_rnn:
            hidden = 1 if arch == "rnn" else rnn_dim
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
            head_in = hidden
        else:  # cnn_nn: flatten conv output
            head_in = filters * (n_timepoints - window + 1)
        if self.use_demo:
            head_in += n_demo
        if arch == "rnn":
            self.head = nn.Identity()
        elif arch == "rnn_nn":
            self.head = nn.Sequential(nn.Linear(head_in, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            self.head = nn.Linear(head_in, 1)

    def forward(self, x, demo=None):
        # x: (batch, T, N)
        if self.use_cnn:
            x = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)   # (B, T', F)
        if self.use_rnn:
            _, (h, _) = self.lstm(x)
            z = h[-1]
        else:
            z = x.flatten(1)
        z = self.drop(z)
        if self.use_demo:
            z = torch.cat([z, demo], dim=1)
        return self.head(z).squeeze(-1)                                    # logits


def should_stop(train_acc: float, dev_acc: float, threshold=0.6, max_acc=0.65, min_acc=0.55):
    """The paper's three exit conditions. Returns (stop, reason)."""
    gap = abs(train_acc - dev_acc)
    tol = (train_acc + dev_acc) / 20
    if gap <= tol and train_acc > threshold and dev_acc > threshold:
        return True, "converged"            # rule 1: close and both above threshold
    if gap >= tol and train_acc > max_acc and train_acc > dev_acc:
        return True, "diverging"            # rule 2: gap growing, train too high
    if train_acc > max_acc and dev_acc < min_acc:
        return True, "overfit"              # rule 3: train high, dev low
    return False, ""


class DeepClassifier:
    def __init__(self, arch: str = "cnn_rnn", epochs: int = 100, batch_size: int = 16,
                 lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.5,
                 min_epochs: int = 5, seed: int = 0, use_rules: bool = True):
        self.arch, self.epochs, self.batch_size = arch, epochs, batch_size
        self.lr, self.weight_decay, self.dropout = lr, weight_decay, dropout
        self.min_epochs, self.seed, self.use_rules = min_epochs, seed, use_rules

    @staticmethod
    def _t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def _accuracy(self, X, D, y):
        return float(((self.predict_proba(X, D)[:, 1] > 0.5) == y).mean())

    def fit(self, X, y, X_dev, y_dev, D=None, D_dev=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_demo = 0 if D is None else D.shape[1]
        # Standardise demographics using training statistics.
        if D is not None:
            self._dmu, self._dsd = D.mean(0), D.std(0) + 1e-8
        self.model = SeqNet(self.arch, X.shape[2], X.shape[1], n_demo, dropout=self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = self._t(X), self._t(y)
        Dt = self._t(self._norm_demo(D)) if D is not None else None
        self.history_ = []
        best = (-1.0, None, 0)
        self.stop_reason_ = "max_epochs"
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            order = rng.permutation(len(y))
            for i in range(0, len(order), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                logits = self.model(Xt[b], Dt[b] if Dt is not None else None)
                loss = loss_fn(logits, yt[b])
                loss.backward()
                opt.step()
            tr_acc = self._accuracy(X, D, y)
            dv_acc = self._accuracy(X_dev, D_dev, y_dev)
            self.history_.append({"epoch": epoch, "train_acc": tr_acc, "dev_acc": dv_acc,
                                  "loss": float(loss)})
            if dv_acc > best[0]:
                best = (dv_acc, copy.deepcopy(self.model.state_dict()), epoch)
            if self.use_rules and epoch >= self.min_epochs:
                stop, reason = should_stop(tr_acc, dv_acc)
                if stop:
                    self.stop_reason_ = reason
                    break
        if self.stop_reason_ in ("max_epochs", "overfit", "diverging"):
            # Fall back to the epoch with the best dev accuracy.
            self.model.load_state_dict(best[1])
            self.best_epoch_ = best[2]
        else:
            self.best_epoch_ = epoch
        self.epochs_run_ = epoch
        return self

    def _norm_demo(self, D):
        return None if D is None else (D - self._dmu) / self._dsd

    @torch.no_grad()
    def predict_proba(self, X, D=None):
        self.model.eval()
        Dt = self._t(self._norm_demo(D)) if D is not None and self.model.use_demo else None
        p = torch.sigmoid(self.model(self._t(X), Dt)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X, D=None):
        return (self.predict_proba(X, D)[:, 1] > 0.5).astype(int)
