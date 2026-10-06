"""Feature engineering from parcellated BOLD time series."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEMOGRAPHIC_COLUMNS = ("age", "sex", "scanner")


def dynamic_range(X: np.ndarray) -> np.ndarray:
    """Paper's derived feature: x_derived(N) = max(x_N) - min(x_N). -> (m, N)"""
    return X.max(axis=1) - X.min(axis=1)


def temporal_std(X: np.ndarray) -> np.ndarray:
    return X.std(axis=1)


def connectivity_matrices(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of regions. -> (m, N, N)"""
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
    return np.einsum("mtn,mtk->mnk", Xc, Xc)


def connectivity(X: np.ndarray) -> np.ndarray:
    """Fisher-z functional connectivity, upper triangle. -> (m, N(N-1)/2)"""
    C = connectivity_matrices(X)
    iu = np.triu_indices(X.shape[2], k=1)
    r = np.clip(C[:, iu[0], iu[1]], -0.999999, 0.999999)
    return np.arctanh(r)


def demographic_matrix(demo: pd.DataFrame, columns=DEMOGRAPHIC_COLUMNS) -> np.ndarray:
    if not columns:
        return np.empty((len(demo), 0))
    return demo[list(columns)].to_numpy(dtype=float)


FEATURE_SETS = {
    "range": dynamic_range,
    "std": temporal_std,
    "conn": connectivity,
}


def build_features(X: np.ndarray, demo: pd.DataFrame, brain: str | None = "range",
                   demographics=DEMOGRAPHIC_COLUMNS) -> np.ndarray:
    """Concatenate a brain feature set (``range``, ``std``, ``conn``,
    ``range+conn`` ... or None) with the chosen demographic columns."""
    parts = []
    if brain:
        for name in brain.split("+"):
            parts.append(FEATURE_SETS[name](X))
    parts.append(demographic_matrix(demo, demographics))
    F = np.concatenate(parts, axis=1)
    if F.shape[1] == 0:
        raise ValueError("Empty feature set")
    return F
