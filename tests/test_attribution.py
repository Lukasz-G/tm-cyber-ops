"""Efficiency of every model's attributions: phi[i].sum() == margin(x_i) - mean margin(background).

Also the closed-form TM Shapley values against brute-force enumeration (tests/shapley_closed_form.jl).
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from tmcyberops.models import LogRegDetector, TsetlinDetector, XGBoostDetector
from tmcyberops.paths import JULIA_PROJECT, REPO, julia_exe


def has_julia() -> bool:
    try:
        julia_exe()
        return True
    except RuntimeError:
        return False


needs_julia = pytest.mark.skipif(not has_julia(), reason="julia not installed")


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    X = rng.random((600, 40)) < 0.2
    y = (X[:, 1] & ~X[:, 4]) | (X[:, 7] & X[:, 9])
    return X, y, X[:20], X[100:110]


def efficiency_gap(model, X, y, explain, background) -> float:
    model.fit(X, y)
    phi = model.attribute(explain, background)
    expected = model.margin(explain) - model.margin(background).mean()
    assert phi.shape == explain.shape
    return float(np.max(np.abs(phi.sum(axis=1) - expected)))


def test_logreg_efficiency(toy):
    assert efficiency_gap(LogRegDetector(), *toy) < 1e-9


def test_xgboost_efficiency(toy):
    assert efficiency_gap(XGBoostDetector(n_estimators=50, max_leaves=8), *toy) < 1e-5


def test_xgboost_efficiency_float64(toy):
    X, y, explain, background = toy
    model = XGBoostDetector(n_estimators=50, max_leaves=8).fit(X, y)
    phi = model.attribute(explain, background)
    expected = model.margin_float64(explain) - model.margin_float64(background).mean()
    assert np.max(np.abs(phi.sum(axis=1) - expected)) < 1e-6
    assert np.max(np.abs(model.margin_float64(explain) - model.margin(explain))) < 1e-4


@needs_julia
def test_tm_efficiency(toy, tmp_path):
    tm = TsetlinDetector(clauses_per_class=10, T=8, S=10, L=16, LF=4, epochs=10, seed=1, workdir=tmp_path)
    assert efficiency_gap(tm, *toy) < 1e-9


@needs_julia
def test_tm_predict_is_margin_rule(toy, tmp_path):
    X, y, explain, _ = toy
    tm = TsetlinDetector(clauses_per_class=10, T=8, S=10, L=16, LF=4, epochs=10, seed=1, workdir=tmp_path)
    tm.fit(X, y)
    assert np.array_equal(tm.predict(X), tm.margin(X) > 0)


@needs_julia
def test_tm_closed_form_matches_brute_force():
    proc = subprocess.run(
        [julia_exe(), f"--project={JULIA_PROJECT}", str(REPO / "tests" / "shapley_closed_form.jl")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0 and "EXACT MATCH" in proc.stdout, proc.stdout + proc.stderr[-3000:]
