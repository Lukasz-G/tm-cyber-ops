"""Logistic regression: the linear control arm.

For a linear margin w.x + c the interventional Shapley value of feature i against a background set is
w_i * (x_i - mean of background_i), exactly.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


class LogRegDetector:
    name = "logreg"
    threshold_rule = "predicted probability > 0.5 (log-odds margin > 0)"

    def __init__(self, C: float = 1.0, max_iter: int = 5000):
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRegDetector:
        self.clf.fit(np.asarray(X, np.float64), np.asarray(y, bool))
        return self

    def update(self, X: np.ndarray, y: np.ndarray) -> LogRegDetector:
        """A new detector fitted on (X, y) with the L2 penalty centred on this model's weights.

        It minimises sum log-loss + |w - w_prev|^2 / (2 C), the objective of the original fit with the
        prior moved from zero to the previous weights, so the update keeps what the earlier data
        taught unless the new rows pull it away. The intercept is not penalised, as in the original fit.
        """
        X = np.asarray(X, np.float64)
        s = np.where(np.asarray(y, bool), 1.0, -1.0)
        w0, b0, C = self.clf.coef_[0].copy(), float(self.clf.intercept_[0]), float(self.clf.C)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = theta[:-1], theta[-1]
            z = s * (X @ w + b)
            g = -s * expit(-z)  # derivative of the log-loss with respect to each margin
            loss = float(np.logaddexp(0.0, -z).sum() + np.sum((w - w0) ** 2) / (2 * C))
            return loss, np.r_[X.T @ g + (w - w0) / C, g.sum()]

        res = minimize(objective, np.r_[w0, b0], jac=True, method="L-BFGS-B", options={"maxiter": 5000})
        new = copy.deepcopy(self)
        new.clf.coef_ = res.x[:-1][None, :]
        new.clf.intercept_ = np.array([res.x[-1]])
        return new

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(np.asarray(X, np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.margin(X) > 0

    def attribute(self, X: np.ndarray, background: np.ndarray) -> np.ndarray:
        w = self.clf.coef_[0]
        mu = np.asarray(background, np.float64).mean(axis=0)
        return (np.asarray(X, np.float64) - mu) * w
