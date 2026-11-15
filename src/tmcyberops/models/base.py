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


def precision_recall_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive (malicious) class, in percent (0 when undefined)."""
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return 100 * prec, 100 * rec, 100 * f


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 of the positive (malicious) class, in percent."""
    return precision_recall_f1(y, pred)[2]


def max_f1_cutoff(y: np.ndarray, margin: np.ndarray) -> float:
    """The margin cut-off that maximises F1 on (y, margin), predictions being margin >= cut-off.

    Candidates are the distinct margins; among equal F1 the lowest cut-off wins. This is the rule of
    tm-cyber-lab research/threshold-corpora (pick_threshold, commit 277531a), written without
    scikit-learn.
    """
    y, margin = np.asarray(y, bool), np.asarray(margin, np.float64)
    order = np.argsort(-margin, kind="stable")
    m, yy = margin[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(m) != 0), len(m) - 1]  # last row of each distinct margin
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    f = 2 * tp / np.maximum(1, tp + fp + int(yy.sum()))  # 2 tp / (2 tp + fp + fn)
    best = np.flatnonzero(f == f.max())[-1]  # descending margins: the last maximum is the lowest cut-off
    return float(m[last[best]])
