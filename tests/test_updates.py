"""Incremental updates of each detector: the old model is left unchanged and the new one learns the new rows."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from tmcyberops.drift import importance, jaccard, pick, ranks, topk
from tmcyberops.models import LogRegDetector, TsetlinDetector, XGBoostDetector
from tmcyberops.paths import julia_exe


def has_julia() -> bool:
    try:
        julia_exe()
        return True
    except RuntimeError:
        return False


@pytest.fixture(scope="module")
def shifted():
    """Old rows follow one rule, new rows another."""
    rng = np.random.default_rng(11)
    X_old = rng.random((800, 30)) < 0.2
    y_old = X_old[:, 1] & ~X_old[:, 4]
    X_new = rng.random((800, 30)) < 0.2
    y_new = X_new[:, 7] | X_new[:, 9]
    return X_old, y_old, X_new, y_new


def logloss(model, X, y) -> float:
    z = np.where(y, 1.0, -1.0) * model.margin(X)
    return float(np.logaddexp(0.0, -z).mean())


def test_logreg_update_is_the_prior_centred_optimum(shifted):
    X_old, y_old, X_new, y_new = shifted
    old = LogRegDetector().fit(X_old, y_old)
    w_before = old.clf.coef_.copy()
    new = old.update(X_new, y_new)
    assert np.array_equal(old.clf.coef_, w_before)
    assert logloss(new, X_new, y_new) < logloss(old, X_new, y_new)
    # same objective solved without the analytic gradient
    s, w0, C = np.where(y_new, 1.0, -1.0), w_before[0], old.clf.C
    Xf = X_new.astype(float)

    def f(t):
        return np.logaddexp(0.0, -s * (Xf @ t[:-1] + t[-1])).sum() + np.sum((t[:-1] - w0) ** 2) / (2 * C)

    ref = minimize(f, np.r_[w0, old.clf.intercept_[0]], method="L-BFGS-B", options={"maxiter": 20000})
    assert f(np.r_[new.clf.coef_[0], new.clf.intercept_[0]]) <= ref.fun + 1e-4


def test_logreg_update_with_a_strong_prior_stays_put(shifted):
    X_old, y_old, X_new, y_new = shifted
    old = LogRegDetector(C=1e-4).fit(X_old, y_old)
    new = old.update(X_new, y_new)
    assert np.max(np.abs(new.clf.coef_ - old.clf.coef_)) < 0.05


def test_xgboost_update_adds_trees(shifted):
    X_old, y_old, X_new, y_new = shifted
    old = XGBoostDetector(n_estimators=30, max_leaves=8).fit(X_old, y_old)
    m_before = old.margin(X_new)
    new = old.update(X_new, y_new, trees=20)
    assert len(new.clf.get_booster().get_dump()) == 50
    assert np.array_equal(old.margin(X_new), m_before)
    assert logloss(new, X_new, y_new) < logloss(old, X_new, y_new)


@pytest.mark.skipif(not has_julia(), reason="julia not installed")
def test_tm_update_continues_from_the_saved_state(shifted, tmp_path):
    X_old, y_old, X_new, y_new = shifted
    old = TsetlinDetector(clauses_per_class=20, S=3, epochs=10, workdir=tmp_path / "old").fit(X_old, y_old)
    m_before = old.margin(X_new)
    new = old.update(X_new, y_new, epochs=10, workdir=tmp_path / "new", rng_offset=100)
    assert new.model_path.exists() and new.model_path != old.model_path
    assert np.array_equal(old.margin(X_new), m_before)
    acc_old = np.mean((old.margin(X_new) > 0) == y_new)
    acc_new = np.mean((new.margin(X_new) > 0) == y_new)
    assert acc_new > acc_old


def test_drift_helpers():
    imp = np.array([0.0, 3.0, 1.0, 3.0, 0.5])
    assert topk(imp, 2).tolist() == [False, True, False, True, False]  # ties by index
    assert topk(imp, 10).sum() == 4  # zero importance never enters
    assert ranks(imp).tolist() == [5, 1, 3, 2, 4]
    assert jaccard(topk(imp, 2), topk(imp, 3)) == pytest.approx(1 / 3)
    assert importance(np.array([[1.0, -2.0], [-3.0, 0.0]])).tolist() == [2.0, 1.0]
    y = np.array([False] * 20 + [True] * 5)
    rows = pick(np.random.default_rng(0), np.arange(25), y, 8, 2)
    assert len(set(rows)) == 10 and y[rows].sum() == 2
