"""The detection metrics and the held-out cut-off rule."""

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from tmcyberops.models import max_f1_cutoff, precision_recall_f1


def sklearn_cutoff(y, s):
    """The reference rule: scikit-learn's precision-recall curve, first maximum of F1."""
    prec, rec, thr = precision_recall_curve(y, s)
    f = np.nan_to_num(2 * prec * rec / np.maximum(1e-12, prec + rec))[:-1]
    return float(thr[int(np.argmax(f))])


@pytest.mark.parametrize("seed", range(20))
def test_cutoff_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(400) < 0.1
    s = np.round(rng.normal(size=400) + 1.5 * y, 1)  # rounding makes ties
    assert max_f1_cutoff(y, s) == sklearn_cutoff(y, s)


def test_cutoff_separable():
    y = np.array([0, 0, 0, 1, 1], bool)
    s = np.array([-2.0, -1.0, 0.5, 1.0, 3.0])
    assert max_f1_cutoff(y, s) == 1.0


def test_precision_recall_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.1
    pred = rng.random(1000) < 0.12
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary")
    np.testing.assert_allclose(precision_recall_f1(y, pred), (100 * p, 100 * r, 100 * f))
