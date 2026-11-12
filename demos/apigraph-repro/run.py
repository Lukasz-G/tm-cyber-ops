# QUESTION:  Does this repository's pipeline reproduce tm-cyber-lab's APIGraph detection numbers, and do
#            the three models' attributions satisfy efficiency on APIGraph rows?
# SCOPE:     APIGraph Drebin features (Chen et al. 2023 extraction), width 1,159; train on the 2012 pool
#            (30,533 rows), test per year 2013-2018. Reference: tm-cyber-lab research/b5-apigraph at commit
#            e6f8fb8, TMCore from tm-lab at 43dba5f.
# ARMS:      FPTM S=100 (20 clauses per class, T=10, L=64, LF=10, 30 epochs, seeds 1-10); XGBoost (5000
#            trees, learning rate 0.02, 256 leaves); logistic regression (linear control, no lab reference).
# PASS/FAIL: PASS if XGBoost's per-year F1 and FPTM's per-year mean and sample sd over the ten seeds equal
#            the lab's to two decimals, and on 100 rows of 2013 against 100 background rows of the pool
#            the attributions sum to margin(x) - mean margin(background) within 1e-9 (FPTM, logistic
#            regression) and 1e-5 (XGBoost). FAIL otherwise.
#            AMENDED 2026-11-12, after the first run and before the second: XGBoost's attributions are
#            checked against a float64 sum of the same trees within 1e-6, because XGBoost's own margin is
#            accumulated in float32 (first run: 3.9e-5 against that margin, 7.8e-8 against the float64
#            sum; see xgb_precision.py). The original check is still computed and reported.
#
#   uv run python demos/apigraph-repro/run.py [--seeds 10]
#
# Threshold rules: FPTM predicts malware when the malware class score exceeds the benign class score
# (ties benign); XGBoost and logistic regression when the predicted probability exceeds 0.5. F1 is the
# malware class, pooled over the months of each test year, as in the lab.

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tmcyberops.datasets import apigraph
from tmcyberops.models import LogRegDetector, TsetlinDetector, XGBoostDetector, f1

HERE = Path(__file__).resolve().parent
YEARS = list(apigraph.TEST_YEARS)

# tm-cyber-lab research/b5-apigraph (commit e6f8fb8): baselines.json and results.txt.
LAB_XGB = {
    2013: 83.28690807799443,
    2014: 69.87875593041645,
    2015: 66.12921539684342,
    2016: 68.39378238341969,
    2017: 72.1267574202187,
    2018: 72.30046948356808,
}
LAB_TM_S100 = {
    2013: (84.24, 2.01),
    2014: (66.76, 3.07),
    2015: (61.25, 2.24),
    2016: (59.65, 3.56),
    2017: (64.55, 4.56),
    2018: (65.28, 4.12),
}
N_EXPLAIN, N_BACKGROUND = 100, 100


def log(msg: str) -> None:
    print(msg, flush=True)


def per_year(pred: np.ndarray, y: np.ndarray, bounds: dict) -> dict:
    return {yr: f1(y[a:b], pred[a:b]) for yr, (a, b) in bounds.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    Xtr, ytr = apigraph.load_train()
    parts = [apigraph.load_year(yr) for yr in YEARS]
    Xte = np.vstack([p[0] for p in parts])
    yte = np.concatenate([p[1] for p in parts])
    bounds, a = {}, 0
    for yr, (_, y) in zip(YEARS, parts, strict=True):
        bounds[yr] = (a, a + len(y))
        a += len(y)
    del parts
    log(
        f"train {Xtr.shape[0]:,} rows x {Xtr.shape[1]:,}, {int(ytr.sum()):,} malware ({100 * ytr.mean():.1f}%)"
    )
    for yr, (a, b) in bounds.items():
        log(f"  {yr}: {b - a:,} rows, {int(yte[a:b].sum()):,} malware ({100 * yte[a:b].mean():.1f}%)")

    rng = np.random.default_rng(0)
    background = Xtr[np.sort(rng.choice(len(Xtr), N_BACKGROUND, replace=False))]
    explain = Xte[np.sort(rng.choice(bounds[2013][1], N_EXPLAIN, replace=False))]

    def efficiency(model) -> float | tuple[float, float]:
        t = time.time()
        phi = model.attribute(explain, background)
        total = phi.sum(axis=1)
        gap = np.abs(total - (model.margin(explain) - model.margin(background).mean()))
        log(
            f"  {model.name} attribution: {N_EXPLAIN} rows x {N_BACKGROUND} background in {time.time() - t:.0f} s,"
            f" max efficiency gap {gap.max():.3g}"
        )
        if not hasattr(model, "margin_float64"):
            return float(gap.max())
        gap64 = np.abs(total - (model.margin_float64(explain) - model.margin_float64(background).mean()))
        log(f"  {model.name} max efficiency gap against the float64 tree sum {gap64.max():.3g}")
        return float(gap.max()), float(gap64.max())

    res: dict = {"seeds": args.seeds, "f1": {}, "efficiency_gap": {}, "time_s": {}}

    log("\nXGBoost")
    t = time.time()
    xgbm = XGBoostDetector().fit(Xtr, ytr)
    res["f1"]["xgboost"] = per_year(xgbm.predict(Xte), yte, bounds)
    res["time_s"]["xgboost_fit_predict"] = round(time.time() - t)
    res["efficiency_gap"]["xgboost"], res["efficiency_gap"]["xgboost_float64"] = efficiency(xgbm)

    log("\nlogistic regression")
    lr = LogRegDetector().fit(Xtr, ytr)
    res["f1"]["logreg"] = per_year(lr.predict(Xte), yte, bounds)
    res["efficiency_gap"]["logreg"] = efficiency(lr)

    log(f"\nFPTM S=100, seeds 1-{args.seeds}")
    tm_runs = []
    for seed in range(1, args.seeds + 1):
        t = time.time()
        tm = TsetlinDetector(S=100, seed=seed, workdir=HERE / "work" / f"seed{seed}").fit(Xtr, ytr)
        scores = per_year(tm.predict(Xte), yte, bounds)
        tm_runs.append(scores)
        log(
            f"  seed {seed}: "
            + " ".join(f"{yr} {v:.2f}" for yr, v in scores.items())
            + f"  ({time.time() - t:.0f} s; {tm.fit_log.splitlines()[-1] if tm.fit_log else ''})"
        )
        if seed == 1:
            res["efficiency_gap"]["tm"] = efficiency(tm)
    res["f1"]["tm_runs"] = tm_runs
    res["f1"]["tm"] = {
        yr: (float(np.mean([r[yr] for r in tm_runs])), float(np.std([r[yr] for r in tm_runs], ddof=1)))
        for yr in YEARS
    }

    log("\n=== per-year F1 (malware), this repository vs tm-cyber-lab b5-apigraph ===")
    log(f"{'year':<6}{'FPTM S=100':>16}{'lab':>16}{'XGBoost':>10}{'lab':>10}{'logreg':>9}")
    ok_xgb = ok_tm = True
    for yr in YEARS:
        m, s = res["f1"]["tm"][yr]
        lm, ls = LAB_TM_S100[yr]
        xg = res["f1"]["xgboost"][yr]
        ok_xgb &= round(xg, 2) == round(LAB_XGB[yr], 2)
        ok_tm &= round(m, 2) == lm and round(s, 2) == ls
        log(
            f"{yr:<6}{m:>9.2f}±{s:<5.2f}{lm:>10.2f}±{ls:<5.2f}{xg:>10.2f}{LAB_XGB[yr]:>10.2f}"
            f"{res['f1']['logreg'][yr]:>9.2f}"
        )
    eff = res["efficiency_gap"]
    ok_exact = eff["tm"] < 1e-9 and eff["logreg"] < 1e-9
    ok_eff_original = ok_exact and eff["xgboost"] < 1e-5
    ok_eff = ok_exact and eff["xgboost_float64"] < 1e-6
    verdict = "PASS" if (ok_xgb and ok_tm and ok_eff) else "FAIL"
    verdict_original = "PASS" if (ok_xgb and ok_tm and ok_eff_original) else "FAIL"
    res.update(
        xgboost_matches_lab=bool(ok_xgb),
        tm_matches_lab=bool(ok_tm),
        efficiency_ok=bool(ok_eff),
        efficiency_ok_original=bool(ok_eff_original),
        verdict=verdict,
        verdict_original_criterion=verdict_original,
        platform=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    )
    log(f"\nXGBoost matches lab: {ok_xgb}   FPTM matches lab: {ok_tm}   efficiency: {ok_eff}")
    log(f"original criterion (XGBoost float32 margin, 1e-5): {verdict_original}")
    log(f"RESULT apigraph_repro={verdict}")
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
