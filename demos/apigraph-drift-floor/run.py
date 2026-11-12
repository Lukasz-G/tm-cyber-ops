# QUESTION:  On APIGraph 2013-2018, in which months does a fixed model's attribution ranking move further
#            than its own same-month noise floor, for FPTM, XGBoost and logistic regression, and does a
#            month-shuffled control stay below that floor?
# SCOPE:     APIGraph Drebin features (Chen et al. 2023 extraction), width 1,159. Models trained once on
#            the 2012 pool (30,533 rows) and kept fixed; 72 monthly test files 2013-01..2018-12; 100
#            explained rows per month against 100 background rows of the pool. Drift as in tm-cyber-lab
#            research/b2-drift (commit e6f8fb8): importance = mean |phi| over the explained rows, the
#            top-k features with nonzero importance, Jaccard distance 1 - |A and B| / |A or B|.
# ARMS:      FPTM S=100 (20 clauses per class, T=10, L=64, LF=10, 30 epochs) seeds 1-3; XGBoost (5000
#            trees, learning rate 0.02, 256 leaves); logistic regression (linear third arm). Control with
#            no temporal signal: the same explained rows shuffled across months, 20 shuffles.
# PASS/FAIL: Floor = same-month distance between two disjoint 100-row halves explained against two
#            disjoint 100-row backgrounds, 50 random splits in each of 12 months (January and July of
#            2013-2018). A month-to-month transition alerts when its k=100 distance exceeds the floor's
#            95th percentile. PASS if the shuffled control alerts in at most 10% of the 71 transitions
#            (mean over shuffles) for every model and every FPTM seed; FAIL otherwise. Real alerts are
#            reported, not gated; k=1000 and a row-only floor are reported as context.
#
#   uv run python demos/apigraph-drift-floor/run.py
#
# Threshold rules for the monthly F1 printed beside each alert: FPTM predicts malware when the malware
# class score exceeds the benign class score (ties benign); XGBoost and logistic regression when the
# predicted probability exceeds 0.5. F1 is the malware class within one month.

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tmcyberops.datasets import apigraph
from tmcyberops.models import LogRegDetector, TsetlinDetector, XGBoostDetector, f1

HERE = Path(__file__).resolve().parent
MONTHS = [f"{y}-{m:02d}" for y in apigraph.TEST_YEARS for m in range(1, 13)]
FLOOR_MONTHS = [f"{y}-{m:02d}" for y in apigraph.TEST_YEARS for m in (1, 7)]
N_ROWS = 100  # explained rows per month, and per half in the floor
N_BG = 100  # background rows
N_SPLITS = 50  # random half splits per floor month
N_SHUFFLES = 20  # month-shuffled control
KS = (100, 1000)
GATE_K = 100
QUANTILE = 0.95
MAX_CONTROL_RATE = 0.10
TM_SEEDS = (1, 2, 3)


def log(msg: str) -> None:
    print(msg, flush=True)


def topk(imp: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k most important features among those with nonzero importance (ties by index)."""
    nz = np.flatnonzero(imp > 0)
    if len(nz) > k:
        nz = nz[np.argsort(-imp[nz], kind="stable")[:k]]
    mask = np.zeros(imp.shape[0], bool)
    mask[nz] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    return 0.0 if union == 0 else 1.0 - int(np.sum(a & b)) / union


def chunked_predict(model, X: np.ndarray, bounds: dict) -> np.ndarray:
    return np.concatenate([model.predict(X[a:b]) for a, b in bounds.values()])


def analyse(name: str, PA: np.ndarray, PB: np.ndarray) -> dict:
    """Drift series, floor, alerts and shuffled control for one model.

    PA: attributions against background 1, rows = 100 per month for the 72 months, then the 100 extra
    rows of each floor month. PB: attributions against background 2, 200 rows per floor month (its 100
    series rows, then its 100 extra rows).
    """
    absA, absB = np.abs(PA), np.abs(PB)
    n_series = len(MONTHS) * N_ROWS
    imp = absA[:n_series].reshape(len(MONTHS), N_ROWS, -1).mean(axis=1)
    out: dict = {"support_mean": float(np.mean(np.sum(imp > 0, axis=1))), "k": {}}
    for k in KS:
        tops = [topk(v, k) for v in imp]
        drift = np.array([jaccard(tops[i - 1], tops[i]) for i in range(1, len(MONTHS))])

        split_rng = np.random.default_rng(1)  # identical splits for every model and k
        floor_rb, floor_r, floor_b = [], [], []
        for j, s in enumerate(FLOOR_MONTHS):
            i = MONTHS.index(s)
            rows_a = np.concatenate(
                [
                    absA[i * N_ROWS : (i + 1) * N_ROWS],
                    absA[n_series + j * N_ROWS : n_series + (j + 1) * N_ROWS],
                ]
            )
            rows_b = absB[j * 2 * N_ROWS : (j + 1) * 2 * N_ROWS]
            for _ in range(N_SPLITS):
                p = split_rng.permutation(2 * N_ROWS)
                h1, h2 = p[:N_ROWS], p[N_ROWS:]
                t1 = topk(rows_a[h1].mean(axis=0), k)
                floor_rb.append(jaccard(t1, topk(rows_b[h2].mean(axis=0), k)))
                floor_r.append(jaccard(t1, topk(rows_a[h2].mean(axis=0), k)))
                floor_b.append(jaccard(t1, topk(rows_b[h1].mean(axis=0), k)))
        thr = float(np.quantile(floor_rb, QUANTILE))
        thr_r = float(np.quantile(floor_r, QUANTILE))

        shuffle_rng = np.random.default_rng(2)
        rates = []
        for _ in range(N_SHUFFLES):
            perm = shuffle_rng.permutation(n_series)
            imp_s = absA[:n_series][perm].reshape(len(MONTHS), N_ROWS, -1).mean(axis=1)
            tops_s = [topk(v, k) for v in imp_s]
            d_s = np.array([jaccard(tops_s[i - 1], tops_s[i]) for i in range(1, len(MONTHS))])
            rates.append(float(np.mean(d_s > thr)))

        alerts = [MONTHS[i + 1] for i in np.flatnonzero(drift > thr)]
        out["k"][k] = {
            "floor_p50": float(np.median(floor_rb)),
            "floor_p95": thr,
            "floor_row_only_p50": float(np.median(floor_r)),
            "floor_row_only_p95": thr_r,
            "floor_background_only_mean": float(np.mean(floor_b)),
            "drift": drift.tolist(),
            "drift_mean": float(drift.mean()),
            "alerts": alerts,
            "alerts_row_only_floor": [MONTHS[i + 1] for i in np.flatnonzero(drift > thr_r)],
            "control_rate_mean": float(np.mean(rates)),
            "control_rate_max": float(np.max(rates)),
        }
        log(
            f"  {name} k={k}: support {out['support_mean']:.0f}, drift mean {drift.mean():.3f}, floor p50 "
            f"{np.median(floor_rb):.3f} p95 {thr:.3f} (row-only p95 {thr_r:.3f}), alerts {len(alerts)}/71, "
            f"control {np.mean(rates):.3f} (max {np.max(rates):.3f})"
        )
    out["imp_tops"] = imp  # kept in memory for the seed spread, dropped before writing
    return out


def main() -> int:
    t0 = time.time()
    Xtr, ytr = apigraph.load_train()
    parts = [apigraph.load(s) for s in MONTHS]
    Xte = np.vstack([p[0] for p in parts])
    yte = np.concatenate([p[1] for p in parts])
    bounds, a = {}, 0
    for s, (_, y) in zip(MONTHS, parts, strict=True):
        bounds[s] = (a, a + len(y))
        a += len(y)
    del parts
    log(f"train {Xtr.shape[0]:,} x {Xtr.shape[1]:,}; test {Xte.shape[0]:,} rows in {len(MONTHS)} months")

    rng = np.random.default_rng(0)
    bg = rng.choice(len(Xtr), 2 * N_BG, replace=False)
    B1, B2 = Xtr[np.sort(bg[:N_BG])], Xtr[np.sort(bg[N_BG:])]
    series, extra = {}, {}
    for s in MONTHS:
        lo, hi = bounds[s]
        idx = lo + rng.choice(hi - lo, 2 * N_ROWS if s in FLOOR_MONTHS else N_ROWS, replace=False)
        series[s] = np.sort(idx[:N_ROWS])
        if s in FLOOR_MONTHS:
            extra[s] = np.sort(idx[N_ROWS:])
    XA = np.vstack([Xte[series[s]] for s in MONTHS] + [Xte[extra[s]] for s in FLOOR_MONTHS])
    XB = np.vstack([Xte[np.concatenate([series[s], extra[s]])] for s in FLOOR_MONTHS])
    log(f"explained: {len(XA):,} rows against background 1, {len(XB):,} against background 2")

    def monthly_f1(pred: np.ndarray) -> dict:
        return {s: f1(yte[a:b], pred[a:b]) for s, (a, b) in bounds.items()}

    res: dict = {"models": {}, "time_s": {}}
    imps: dict = {}

    def run(name: str, model, pred_fn) -> None:
        t = time.time()
        log(f"\n{name}")
        model.fit(Xtr, ytr)
        f1m = monthly_f1(pred_fn(model))
        PA = model.attribute(XA, B1)
        PB = model.attribute(XB, B2)
        log(f"  fit, predict, attribute {len(XA) + len(XB):,} rows in {time.time() - t:.0f} s")
        r = analyse(name, PA, PB)
        imps[name] = r.pop("imp_tops")
        r.update(threshold_rule=model.threshold_rule, f1_month=f1m)
        res["models"][name] = r
        res["time_s"][name] = round(time.time() - t)

    run("logreg", LogRegDetector(), lambda m: chunked_predict(m, Xte, bounds))
    for seed in TM_SEEDS:
        tm = TsetlinDetector(S=100, seed=seed, workdir=HERE / "work" / f"seed{seed}")
        run(f"fptm-seed{seed}", tm, lambda m: m.predict(Xte))
    run("xgboost", XGBoostDetector(), lambda m: chunked_predict(m, Xte, bounds))

    # Context: same month, same rows, same background, different FPTM seed.
    spread = {}
    for k in KS:
        vals = [
            jaccard(topk(imps[f"fptm-seed{a}"][i], k), topk(imps[f"fptm-seed{b}"][i], k))
            for a in TM_SEEDS
            for b in TM_SEEDS
            if a < b
            for i in range(len(MONTHS))
        ]
        spread[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "n": len(vals)}
    res["fptm_seed_spread"] = spread
    log(f"\nFPTM same-month seed spread: {', '.join(f'k={k} {v["mean"]:.3f}' for k, v in spread.items())}")

    with (HERE / "series.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model", "k", "from", "to", "drift", "floor_p95", "alert", "f1_from", "f1_to"])
        for name, r in res["models"].items():
            for k in KS:
                rk = r["k"][k]
                for i, d in enumerate(rk["drift"]):
                    a, b = MONTHS[i], MONTHS[i + 1]
                    w.writerow(
                        [
                            name,
                            k,
                            a,
                            b,
                            f"{d:.4f}",
                            f"{rk['floor_p95']:.4f}",
                            int(d > rk["floor_p95"]),
                            f"{r['f1_month'][a]:.2f}",
                            f"{r['f1_month'][b]:.2f}",
                        ]
                    )

    log(f"\n=== gate: shuffled control alert rate at k={GATE_K} (max {MAX_CONTROL_RATE:.0%}) ===")
    ok = True
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        ok &= rk["control_rate_mean"] <= MAX_CONTROL_RATE
        log(
            f"{name:<13} control {rk['control_rate_mean']:.3f}  alerts {len(rk['alerts']):>2}/71  "
            f"floor p95 {rk['floor_p95']:.3f}  drift mean {rk['drift_mean']:.3f}"
        )
    log("\n=== alerts at k=100 (month: drift / floor p95, F1 previous -> this month) ===")
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        items = [
            f"{s} {rk['drift'][MONTHS.index(s) - 1]:.3f}/{rk['floor_p95']:.3f} "
            f"({r['f1_month'][MONTHS[MONTHS.index(s) - 1]]:.1f}->{r['f1_month'][s]:.1f})"
            for s in rk["alerts"]
        ]
        log(f"{name}: " + ("; ".join(items) if items else "none"))

    verdict = "PASS" if ok else "FAIL"
    res.update(
        verdict=verdict,
        config=dict(
            months=MONTHS,
            floor_months=FLOOR_MONTHS,
            n_rows=N_ROWS,
            n_bg=N_BG,
            n_splits=N_SPLITS,
            n_shuffles=N_SHUFFLES,
            ks=KS,
            gate_k=GATE_K,
            quantile=QUANTILE,
            max_control_rate=MAX_CONTROL_RATE,
            tm_seeds=TM_SEEDS,
        ),
        platform=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    )
    res["time_s"]["total"] = round(time.time() - t0)
    log(f"\nRESULT apigraph_drift_floor={verdict}  ({res['time_s']['total']} s)")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
