"""Factories for the feature-based (non deep-learning) models."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .logistic_newton import NewtonLogisticRegression

SVM_KERNELS = {
    "L": dict(kernel="linear"),
    "P": dict(kernel="poly", degree=2),
    "S": dict(kernel="sigmoid"),
    "RB": dict(kernel="rbf"),
}


def newton_lr(l2: float = 0.0):
    """Paper baseline: standardised features -> Newton's-method LR."""
    return make_pipeline(StandardScaler(), NewtonLogisticRegression(l2=l2, tol=1e-5))


def svm(kernel_code: str, tune: bool = True, seed: int = 0):
    """SVM with C (and gamma) chosen by inner 3-fold CV on the training fold."""
    base = make_pipeline(StandardScaler(),
                         SVC(random_state=seed, **SVM_KERNELS[kernel_code]))
    if not tune:
        return base
    grid = {"svc__C": [0.01, 0.1, 1, 10]}
    if kernel_code != "L":
        grid["svc__gamma"] = ["scale", 1e-2, 1e-3]
    return GridSearchCV(base, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        scoring="accuracy", n_jobs=1)


def l2_logistic(seed: int = 0, pca_components: int | None = None,
                class_weight: str | None = None):
    """Regularised LR for high-dimensional connectivity features; the L2
    strength is picked by inner 3-fold CV on the training fold."""
    steps = [StandardScaler()]
    if pca_components:
        steps.append(PCA(n_components=pca_components, random_state=seed))
    steps.append(LogisticRegression(max_iter=5000, solver="lbfgs", class_weight=class_weight))
    pipe = make_pipeline(*steps)
    return GridSearchCV(pipe, {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
                        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        scoring="roc_auc", n_jobs=1)


class LateFusion(ClassifierMixin, BaseEstimator):
    """Late fusion of a demographics model and an fMRI model.

    The last ``n_demo`` columns are demographics; the rest are fMRI features.
    Each block gets its own classifier and the two predicted log-odds are
    averaged, so three demographic columns are not drowned out by hundreds of
    connectivity features under a single shared L2 penalty.
    """

    def __init__(self, n_demo: int = 3, seed: int = 0):
        self.n_demo = n_demo
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.array([0, 1])
        self.demo_model_ = newton_lr().fit(X[:, -self.n_demo:], y)
        self.fmri_model_ = l2_logistic(seed=self.seed).fit(X[:, :-self.n_demo], y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        p = [self.demo_model_.predict_proba(X[:, -self.n_demo:])[:, 1],
             self.fmri_model_.predict_proba(X[:, :-self.n_demo])[:, 1]]
        logit = [np.log(np.clip(q, 1e-9, 1 - 1e-9) / np.clip(1 - q, 1e-9, 1)) for q in p]
        return 0.5 * (logit[0] + logit[1])

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
