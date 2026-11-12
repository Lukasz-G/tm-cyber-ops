"""The interface every model in this repository implements.

All models are binary detectors (true = malicious) on binarised features. Each exposes a real-valued
margin and its own decision rule, stated as a threshold on that margin, and explains the margin with
interventional Shapley values against a background set:

    phi[i].sum() == margin(X[i]) - margin(background).mean()     (efficiency)

That property is checked in the tests for every model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    name: str
    threshold_rule: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> Detector: ...

    def margin(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def attribute(self, X: np.ndarray, background: np.ndarray) -> np.ndarray: ...


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 of the positive (malicious) class, in percent."""
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 100 * 2 * prec * rec / (prec + rec) if prec + rec else 0.0
