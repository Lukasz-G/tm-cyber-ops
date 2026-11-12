"""Where XGBoost's efficiency gap in run.py comes from.

    uv run python demos/apigraph-repro/xgb_precision.py

run.py checks phi.sum() against margin(x) - mean margin(background), with the margin from XGBoost's own
predict, which accumulates 5000 tree outputs in float32. shap's interventional TreeSHAP works in float64.
This script refits the same model (deterministic), takes the same rows, and compares the attribution sums
with both margins: XGBoost's float32 margin and a float64 sum of the same trees (shap's TreeEnsemble).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import shap

from tmcyberops.datasets import apigraph
from tmcyberops.models import XGBoostDetector

HERE = Path(__file__).resolve().parent


def main() -> int:
    Xtr, ytr = apigraph.load_train()
    X13, _ = apigraph.load_year(2013)
    rng = np.random.default_rng(0)  # the same draws, in the same order, as run.py
    background = Xtr[np.sort(rng.choice(len(Xtr), 100, replace=False))]
    explain = X13[np.sort(rng.choice(len(X13), 100, replace=False))]

    model = XGBoostDetector().fit(Xtr, ytr)
    explainer = shap.TreeExplainer(
        model.clf,
        data=background.astype(np.float64),
        feature_perturbation="interventional",
        model_output="raw",
    )
    phi = np.asarray(explainer.shap_values(explain.astype(np.float64), check_additivity=False))
    total = phi.sum(axis=1)

    m32_x, m32_b = model.margin(explain), model.margin(background)
    m64_x = explainer.model.predict(explain.astype(np.float64))
    m64_b = explainer.model.predict(background.astype(np.float64))

    out = {
        "max_abs_margin_float32_minus_float64": float(
            max(np.abs(m32_x - m64_x).max(), np.abs(m32_b - m64_b).max())
        ),
        "efficiency_gap_vs_float32_margin": float(np.abs(total - (m32_x - m32_b.mean())).max()),
        "efficiency_gap_vs_float64_margin": float(np.abs(total - (m64_x - m64_b.mean())).max()),
        "max_abs_margin": float(np.abs(m64_x).max()),
    }
    for k, v in out.items():
        print(f"{k:<40} {v:.3g}")
    (HERE / "xgb_precision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
