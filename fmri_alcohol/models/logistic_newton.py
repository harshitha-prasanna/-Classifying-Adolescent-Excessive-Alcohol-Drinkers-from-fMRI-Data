"""Logistic regression trained with Newton's method (implemented from scratch).

    h_theta(x) = 1 / (1 + exp(-theta^T x))
    l(theta)   = sum_i y_i log h(x_i) + (1 - y_i) log(1 - h(x_i))  -  (lambda/2)||w||^2
    theta     := theta - H^{-1} grad l(theta)

Converges when ||d theta||_2^2 < tol (the paper used tol = 1e-5).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class NewtonLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, l2: float = 0.0, tol: float = 1e-5, max_iter: int = 50):
        self.l2 = l2
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])      # intercept column
        d = Xb.shape[1]
        theta = np.zeros(d)
        reg = np.full(d, self.l2)
        reg[0] = 0.0                                          # don't penalise bias
        self.loss_history_ = []
        for it in range(self.max_iter):
            h = _sigmoid(Xb @ theta)
            grad = Xb.T @ (h - y) + reg * theta               # gradient of -l(theta)
            W = h * (1 - h)
            H = (Xb * W[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d)
            step = np.linalg.solve(H, grad)
            theta -= step
            self.loss_history_.append(self._nll(Xb, y, theta))
            if step @ step < self.tol:
                break
        self.n_iter_ = it + 1
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def _nll(self, Xb, y, theta):
        p = np.clip(_sigmoid(Xb @ theta), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
                     + 0.5 * self.l2 * theta[1:] @ theta[1:])

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
