"""Evaluation protocols and metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .preprocess import balance_undersample, dev_split, kfold_splits


def predict_scores(model, X) -> np.ndarray:
    """P(heavy drinker); falls back to sigmoid(decision_function) for SVMs."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


@dataclass
class ProtocolA:
    """Reference-study protocol: balance classes by undersampling, set aside a
    5% dev set, then 10-fold CV on the rest. Repeated over ``repeats``
    different undersamplings for stable estimates."""
    k: int = 10
    dev_frac: float = 0.05
    repeats: int = 5

    def splits(self, y):
        for r in range(self.repeats):
            bal = balance_undersample(y, seed=r)
            cv_idx, dev_idx = dev_split(bal, y, self.dev_frac, seed=r)
            for f, (tr, te) in enumerate(kfold_splits(cv_idx, y, self.k, seed=r)):
                yield r, f, tr, te, dev_idx


def run_feature_model(name: str, make_model: Callable, F: np.ndarray, y: np.ndarray,
                      protocol: ProtocolA, **tags):
    """Cross-validate an sklearn-style model on a fixed feature matrix.
    Returns (per-fold metrics, per-subject test predictions)."""
    rows, preds = [], []
    t0 = time.time()
    for r, f, tr, te, dev in protocol.splits(y):
        model = make_model()
        model.fit(F[tr], y[tr])
        p_te = predict_scores(model, F[te])
        row = {"model": name, "repeat": r, "fold": f, **tags,
               "train_accuracy": accuracy_score(y[tr], model.predict(F[tr])),
               "dev_accuracy": accuracy_score(y[dev], model.predict(F[dev]))}
        row.update(classification_metrics(y[te], p_te))
        rows.append(row)
        preds.append(pd.DataFrame({"model": name, "repeat": r, "subject": te,
                                   "y": y[te], "p": p_te, **tags}))
    df = pd.DataFrame(rows)
    df["seconds"] = time.time() - t0
    return df, pd.concat(preds, ignore_index=True)


def run_deep_model(name: str, make_model: Callable, X: np.ndarray, D: np.ndarray | None,
                   y: np.ndarray, protocol: ProtocolA, **tags):
    """Returns (per-fold metrics, test predictions, per-epoch training history)."""
    rows, preds, histories = [], [], []
    t0 = time.time()
    for r, f, tr, te, dev in protocol.splits(y):
        model = make_model(seed=1000 * r + f)
        sub = (lambda idx: None) if D is None else (lambda idx: D[idx])
        model.fit(X[tr], y[tr], X[dev], y[dev], sub(tr), sub(dev))
        p_te = model.predict_proba(X[te], sub(te))[:, 1]
        row = {"model": name, "repeat": r, "fold": f, **tags,
               "train_accuracy": accuracy_score(y[tr], model.predict(X[tr], sub(tr))),
               "dev_accuracy": accuracy_score(y[dev], model.predict(X[dev], sub(dev))),
               "epochs": model.epochs_run_, "best_epoch": model.best_epoch_,
               "stop_reason": model.stop_reason_}
        row.update(classification_metrics(y[te], p_te))
        rows.append(row)
        preds.append(pd.DataFrame({"model": name, "repeat": r, "subject": te,
                                   "y": y[te], "p": p_te, **tags}))
        h = pd.DataFrame(model.history_)
        h["model"], h["repeat"], h["fold"] = name, r, f
        histories.append(h)
    df = pd.DataFrame(rows)
    df["seconds"] = time.time() - t0
    return df, pd.concat(preds, ignore_index=True), pd.concat(histories, ignore_index=True)


def run_full_cohort(name: str, make_model: Callable, F: np.ndarray, y: np.ndarray,
                    k: int = 10, repeats: int = 3, **tags) -> pd.DataFrame:
    """Protocol B: all subjects (imbalanced), stratified k-fold, models use
    class weighting instead of discarding data."""
    rows = []
    idx = np.arange(len(y))
    for r in range(repeats):
        for f, (tr, te) in enumerate(kfold_splits(idx, y, k, seed=100 + r)):
            model = make_model()
            model.fit(F[tr], y[tr])
            p_te = predict_scores(model, F[te])
            # Class weights already rebalance the decision boundary, so 0.5 is used.
            row ={"model": name, "repeat": r, "fold": f, **tags}
            row.update(classification_metrics(y[te], p_te))
            rows.append(row)
    return pd.DataFrame(rows)


METRIC_COLS = ["accuracy", "f1", "auc", "precision", "recall", "specificity",
               "balanced_accuracy", "train_accuracy", "dev_accuracy"]


def summarise(df: pd.DataFrame, by=("parcellation", "model")) -> pd.DataFrame:
    by = [c for c in by if c in df.columns]
    cols = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby(by, sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return out.reset_index()
