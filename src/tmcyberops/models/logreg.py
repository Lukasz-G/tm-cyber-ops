"""Logistic regression: the linear control arm.

For a linear margin w.x + c the interventional Shapley value of feature i against a background set is
w_i * (x_i - mean of background_i), exactly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogRegDetector:
    name = "logreg"
    threshold_rule = "predicted probability > 0.5 (log-odds margin > 0)"

    def __init__(self, C: float = 1.0, max_iter: int = 5000):
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRegDetector:
        self.clf.fit(np.asarray(X, np.float64), np.asarray(y, bool))
        return self

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(np.asarray(X, np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.margin(X) > 0

    def attribute(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        w = self.clf.coef_[0]
        mu = np.asarray(background, np.float64).mean(axis=0)
        return (np.asarray(X, np.float64) - mu) * w
