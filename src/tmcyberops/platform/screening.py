"""Package screening on Datadog's malicious PyPI packages and a matched benign set.

    uv run python -m tmcyberops.platform.screening build

The three detectors of the platform, with the platform's settings, are trained on the packages
published before 2023-07 and score every later package. Each is judged per quarter at two rules: its
own default (margin > 0) and the max-F1 cut-off chosen on 10% of the training packages held out
(stratified). Every package after the training period is explained: FPTM in closed form, XGBoost by
interventional TreeSHAP, logistic regression exactly, all against the same 100 training packages
(stratified). The page (screening_page.py) draws those attributions over the code.

Output in work/platform/datadog/: packages.json (one row per package, with each model's margin and
verdicts), metrics.json (per quarter, model and rule), phi-<model>.npy (one row per explained
package, in the order of explained.json), meta.json (split, cut-offs, background, threshold rules).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tmcyberops.datasets import codefeatures
from tmcyberops.models.base import max_f1_cutoff, precision_recall_f1
from tmcyberops.models.logreg import LogRegDetector
from tmcyberops.models.tm import TsetlinDetector
from tmcyberops.models.xgb import XGBoostDetector
from tmcyberops.platform import config

TRAIN_END = "2023-07-01"
MODELS = ("fptm", "xgboost", "logreg")
LABELS = {"fptm": "FPTM", "xgboost": "XGBoost", "logreg": "Logistic regression"}


def out_dir() -> Path:
    p = config.home() / "datadog"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stratified(idx: np.ndarray, y: np.ndarray, share: float | int, rng: np.random.Generator) -> np.ndarray:
    """A stratified draw from idx: a share of each class, or (int) that many rows in the class ratio."""
    pick = []
    total = len(idx)
    for c in (0, 1):
        cls = idx[y[idx] == c]
        k = round(share * len(cls)) if isinstance(share, float) else round(share * len(cls) / total)
        pick.append(rng.choice(cls, size=max(1, k), replace=False))
    return np.sort(np.concatenate(pick))


def detector(model: str, cfg: dict):
    if model == "fptm":
        return TsetlinDetector(seed=cfg["fptm_seed"], workdir=out_dir() / "fptm")
    if model == "xgboost":
        return XGBoostDetector(shap_jobs=cfg["xgboost_shap_jobs"])
    return LogRegDetector()


def build(seed: int = 0) -> None:
    cfg = config.load()
    X, rows, feats = codefeatures.load()
    y = np.array([r["label"] for r in rows], np.int8)
    rng = np.random.default_rng(seed)
    train = np.flatnonzero([r["date"] < TRAIN_END for r in rows])
    test = np.flatnonzero([r["date"] >= TRAIN_END for r in rows])
    held = _stratified(train, y, 0.1, rng)
    fit = np.setdiff1d(train, held)
    background = _stratified(fit, y, 100, rng)
    print(f"train {len(fit)} (+{len(held)} held out), test {len(test)}, features {len(feats)}", flush=True)

    meta = dict(train_end=TRAIN_END, fit=len(fit), held=len(held), test=len(test), seed=seed,
                background=[rows[i]["id"] for i in background], cutoff={}, rule={}, seconds={})  # fmt: skip
    phi_rows = test
    for model in MODELS:
        t0 = time.time()
        det = detector(model, cfg).fit(X[fit], y[fit])
        margin = det.margin(X)
        cut = max_f1_cutoff(y[held], margin[held])
        for i, r in enumerate(rows):
            r[f"{model}_margin"] = float(margin[i])
            r[f"{model}_default"] = bool(margin[i] > 0)
            r[f"{model}_cutoff"] = bool(margin[i] >= cut)
        phi = det.attribute(X[phi_rows], X[background])
        np.save(out_dir() / f"phi-{model}.npy", phi.astype(np.float32))
        meta["cutoff"][model] = float(cut)
        meta["rule"][model] = det.threshold_rule
        meta["seconds"][model] = round(time.time() - t0, 1)
        print(f"  {model}: cut-off {cut:.4f}, {meta['seconds'][model]} s", flush=True)

    for i in held:
        rows[i]["split"] = "held-out"
    for i in fit:
        rows[i]["split"] = "fit"
    for i in test:
        rows[i]["split"] = "test"
    metrics = []
    groups = {"held-out (training period)": held}
    for q in sorted({rows[i]["quarter"] for i in test}):
        groups[q] = np.array([i for i in test if rows[i]["quarter"] == q])
    for name, idx in groups.items():
        for model in MODELS:
            for rule in ("default", "cutoff"):
                pred = np.array([rows[i][f"{model}_{rule}"] for i in idx], bool)
                p, r_, f = precision_recall_f1(y[idx].astype(bool), pred)
                metrics.append(dict(period=name, model=model, rule=rule, precision=p, recall=r_, f1=f,
                                    malicious=int(y[idx].sum()), benign=int(len(idx) - y[idx].sum())))  # fmt: skip
    d = out_dir()
    (d / "packages.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=0), encoding="utf-8")
    (d / "explained.json").write_text(json.dumps([rows[i]["id"] for i in phi_rows]), encoding="utf-8")
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for m in metrics:
        if m["rule"] == "cutoff":
            print(
                f"  {m['period']:28s} {m['model']:8s} F1 {m['f1']:5.1f}  ({m['malicious']} mal, {m['benign']} ben)"
            )
    print("done:", d)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
