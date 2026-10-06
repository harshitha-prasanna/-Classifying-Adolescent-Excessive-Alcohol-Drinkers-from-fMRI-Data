"""Pre-processing: normalisation, class balancing and data splits."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def zscore_timeseries(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score every region of every subject over time. X: (m, T, N)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)


def balance_undersample(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices of a 50/50 subset: all minority samples + an equal random
    sample of the majority class (the paper's "downscaling")."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
    keep = rng.choice(majority, size=len(minority), replace=False)
    return np.sort(np.concatenate([minority, keep]))


def dev_split(idx: np.ndarray, y: np.ndarray, frac: float = 0.05, seed: int = 0):
    """Hold out a small stratified dev set (never used in cross-validation)."""
    cv_idx, dev_idx = train_test_split(idx, test_size=frac, stratify=y[idx],
                                       random_state=seed)
    return np.sort(cv_idx), np.sort(dev_idx)


def kfold_splits(idx: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 0):
    """Stratified k-fold over ``idx``; yields (train_idx, test_idx) in
    absolute subject indices."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(idx, y[idx]):
        yield idx[tr], idx[te]
