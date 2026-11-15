# QUESTION:  On APIGraph 2013-2018, with 500 explained rows per month at a fixed class mix and a noise
#            floor taken under the same fixed background as the series, does a month-shuffled control
#            alert about as often as the floor itself, and in which months does a fixed model's
#            attribution ranking move above that floor, month to month and against a held-out reference
#            from its training year, for FPTM, XGBoost and logistic regression?
# SCOPE:     APIGraph Drebin features (Chen et al. 2023 extraction), width 1,159. The 2012 pool (30,533
#            rows, 10.0% malware) split by a label-stratified random draw: models train once on 27,480
#            rows at that native share, with no class weights and no resampling, and stay fixed; 3,053
#            rows (10% of each class) are held out. 72 monthly test files 2013-01..2018-12 (5.8-10.1%
#            malware). Every explained set has the pool's mix, 450 clean and 50 malware rows of each 500:
#            each month, each floor half, the reference and each reference-floor draw. One 100-row
#            background (90 clean, 10 malware) drawn from the training rows (the deployment background).
#            Drift measure as in tm-cyber-lab research/b2-drift, class mix as in research/b2-corpora,
#            threshold rule as in research/threshold-corpora (all at commit 277531a): importance = mean
#            |phi| over the explained rows, the top-k features with nonzero importance, Jaccard distance
#            1 - |A and B| / |A or B|.
# ARMS:      FPTM S=100 (20 clauses per class, T=10, L=64, LF=10, 30 epochs) seeds 1-10; XGBoost (5000
#            trees, learning rate 0.02, 256 leaves); logistic regression (linear third arm). Control with
#            no temporal order: the 36,000 explained rows shuffled across months within each class, so
#            every shuffled month keeps 450 clean and 50 malware rows, 20 shuffles.
# PASS/FAIL: Floor = same-month distance between two disjoint halves (450 clean and 50 malware each),
#            both under the deployment background, 50 random splits in each of 24 months (January, April,
#            July and October of 2013-2018). A month-to-month transition alerts when its k=100 distance
#            exceeds the floor's 95th percentile. The floor's own exceedance rate e is the share of floor
#            distances above that cut-off (at most 5%, less where distances tie on it). PASS if, for every
#            model and every FPTM seed, e > 0 and the shuffled control alerts in between e/2 and 2e of
#            the 71 transitions (mean over shuffles); FAIL otherwise. Reported, not gated: the
#            month-to-month alerts; each month's distance from the reference (500 held-out rows) against a
#            reference floor (the same 500 rows against 1,200 draws of 500 other held-out rows, 95th
#            percentile); each month's distance from 500 rows of the other 11 months of its own year,
#            against the same reference floor (how far one month sits from a year-wide sample without the
#            gap back to 2012); k=1000; the same rows under a second background; the FPTM seed spread;
#            detection.
#
#   uv run python demos/apigraph-drift-reference/run.py [--reuse]
#
# --reuse loads each model's attributions and margins from work/phi/ (written by an earlier run on the
# same sample) instead of refitting and recomputing them.
#
# Detection, per month and on the held-out rows: precision, recall and F1 of the malware class at two
# cut-offs on each model's margin. (1) The model's default rule, margin > 0: FPTM predicts malware when
# the malware class score exceeds the benign class score (ties benign); XGBoost and logistic regression
# when the predicted probability exceeds 0.5. (2) The cut-off that maximises F1 on the 3,053 held-out
# 2012 rows (10.0% malware, the training share), applied as margin >= cut-off; under (2) the held-out
# figures are measured on the rows that chose the cut-off.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tmcyberops.datasets import apigraph
from tmcyberops.models import (
    LogRegDetector,
    TsetlinDetector,
    XGBoostDetector,
    max_f1_cutoff,
    precision_recall_f1,
)

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
MONTHS = [f"{y}-{m:02d}" for y in apigraph.TEST_YEARS for m in range(1, 13)]
FLOOR_MONTHS = [f"{y}-{m:02d}" for y in apigraph.TEST_YEARS for m in (1, 4, 7, 10)]
YEARS = [str(y) for y in apigraph.TEST_YEARS]
N_ROWS = 500  # explained rows per month, per floor half, in the reference and per reference-floor draw
N_MAL = 50  # malware rows of those 500 (the 2012 pool's 10.0%)
N_BG = 100  # background rows
N_BG_MAL = 10  # malware rows of the background
HOLDOUT = 0.10  # share of each class of the 2012 pool held out
N_SPLITS = 50  # random half splits per floor month
N_REF_DRAWS = 1200  # draws of other held-out rows for the reference floor
N_SHUFFLES = 20  # month-shuffled control
KS = (100, 1000)
GATE_K = 100
QUANTILE = 0.95
CONTROL_FACTOR = 2.0  # control alert rate within [e / 2, 2 e], e = the floor's own exceedance rate
TM_SEEDS = tuple(range(1, 11))
SHAP_JOBS = 12  # processes for XGBoost's TreeSHAP; per-row results do not depend on it
PREDICT_CHUNK = 20_000
RULES = ("default", "held-out")


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


def chunked_margin(model, X: np.ndarray) -> np.ndarray:
    return np.concatenate([model.margin(X[a : a + PREDICT_CHUNK]) for a in range(0, len(X), PREDICT_CHUNK)])


def pick(rng: np.random.Generator, idx: np.ndarray, y: np.ndarray, n_clean: int, n_mal: int):
    """n_clean clean and n_mal malware rows drawn from idx without replacement."""
    return (
        rng.choice(idx[~y[idx]], n_clean, replace=False),
        rng.choice(idx[y[idx]], n_mal, replace=False),
    )


def draw_sample(ytr: np.ndarray, yte: np.ndarray, bounds: dict) -> dict:
    """Every random choice of rows, from one generator (seed 0)."""
    rng = np.random.default_rng(0)
    nc, nm = N_ROWS - N_MAL, N_MAL
    hold = np.sort(
        np.concatenate(
            [
                rng.permutation(np.flatnonzero(ytr == c))[: round(HOLDOUT * int(np.sum(ytr == c)))]
                for c in (False, True)
            ]
        )
    )
    fit = np.setdiff1d(np.arange(len(ytr)), hold)
    bc, bm = N_BG - N_BG_MAL, N_BG_MAL
    c, m = pick(rng, fit, ytr, 2 * bc, 2 * bm)
    smp = {
        "hold": hold,
        "fit": fit,
        "bg1": np.sort(np.r_[c[:bc], m[:bm]]),
        "bg2": np.sort(np.r_[c[bc:], m[bm:]]),
    }
    smp["ref"] = np.sort(np.concatenate(pick(rng, hold, ytr, nc, nm)))
    smp["rest"] = np.setdiff1d(hold, smp["ref"])
    for s in MONTHS:
        lo, hi = bounds[s]
        twice = 2 if s in FLOOR_MONTHS else 1
        c, m = pick(rng, np.arange(lo, hi), yte, twice * nc, twice * nm)
        smp[f"series/{s}"] = np.sort(np.r_[c[:nc], m[:nm]])
        if s in FLOOR_MONTHS:
            smp[f"extra/{s}"] = np.sort(np.r_[c[nc:], m[nm:]])
    return smp


def sample_hash(smp: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(smp):
        h.update(key.encode())
        h.update(np.ascontiguousarray(smp[key], np.int64).tobytes())
    return h.hexdigest()


def split_by_class(rng: np.random.Generator, labels: np.ndarray, n_groups: int) -> np.ndarray:
    """Positions of `labels` dealt into n_groups groups, each with the same counts of clean and malware rows."""
    c = rng.permutation(np.flatnonzero(~labels)).reshape(n_groups, -1)
    m = rng.permutation(np.flatnonzero(labels)).reshape(n_groups, -1)
    return np.concatenate([c, m], axis=1)


def analyse(name: str, PA: np.ndarray, PB: np.ndarray, yA: np.ndarray) -> dict:
    """Both drift series, their floors, alerts and the shuffled control for one model.

    PA, against the deployment background: 500 series rows for each of the 72 months, then the 500 extra
    rows of each floor month, then the 500 reference rows, then the other held-out rows; yA their labels.
    PB, against the second background: the 500 series rows of each floor month.
    """
    absA, absB = np.abs(PA), np.abs(PB)
    nm, nf = len(MONTHS), len(FLOOR_MONTHS)
    n_series = nm * N_ROWS
    off_ref = n_series + nf * N_ROWS
    off_rest = off_ref + N_ROWS
    imp = absA[:n_series].reshape(nm, N_ROWS, -1).mean(axis=1)
    imp_ref = absA[off_ref:off_rest].mean(axis=0)
    rest, y_rest = absA[off_rest:], yA[off_rest:]
    rest_c, rest_m = np.flatnonzero(~y_rest), np.flatnonzero(y_rest)
    out: dict = {
        "support_mean": float(np.mean(np.sum(imp > 0, axis=1))),
        "support_reference": int(np.sum(imp_ref > 0)),
        "k": {},
    }
    for k in KS:
        tops = [topk(v, k) for v in imp]
        drift = np.array([jaccard(tops[i - 1], tops[i]) for i in range(1, nm)])

        split_rng = np.random.default_rng(1)  # identical splits for every model and k
        floor, bg_only = [], []
        for j, s in enumerate(FLOOR_MONTHS):
            i = MONTHS.index(s)
            ext = slice(n_series + j * N_ROWS, n_series + (j + 1) * N_ROWS)
            rows = np.vstack([absA[i * N_ROWS : (i + 1) * N_ROWS], absA[ext]])
            labels = np.r_[yA[i * N_ROWS : (i + 1) * N_ROWS], yA[ext]]
            for _ in range(N_SPLITS):
                h1, h2 = split_by_class(split_rng, labels, 2)
                floor.append(jaccard(topk(rows[h1].mean(axis=0), k), topk(rows[h2].mean(axis=0), k)))
            bg_only.append(jaccard(tops[i], topk(absB[j * N_ROWS : (j + 1) * N_ROWS].mean(axis=0), k)))
        thr = float(np.quantile(floor, QUANTILE))
        exceed = float(np.mean(np.array(floor) > thr))

        shuffle_rng = np.random.default_rng(2)
        rates, control_d = [], []
        for _ in range(N_SHUFFLES):
            groups = split_by_class(shuffle_rng, yA[:n_series], nm)
            tops_s = [topk(absA[g].mean(axis=0), k) for g in groups]
            d_s = np.array([jaccard(tops_s[i - 1], tops_s[i]) for i in range(1, nm)])
            rates.append(float(np.mean(d_s > thr)))
            control_d.append(d_s)
        rate = float(np.mean(rates))
        gate = exceed > 0 and exceed / CONTROL_FACTOR <= rate <= CONTROL_FACTOR * exceed

        top_ref = topk(imp_ref, k)
        ref_drift = np.array([jaccard(top_ref, t) for t in tops])
        ref_rng = np.random.default_rng(3)
        ref_floor = []
        for _ in range(N_REF_DRAWS):
            draw = np.r_[
                ref_rng.choice(rest_c, N_ROWS - N_MAL, replace=False),
                ref_rng.choice(rest_m, N_MAL, replace=False),
            ]
            ref_floor.append(jaccard(top_ref, topk(rest[draw].mean(axis=0), k)))
        thr_ref = float(np.quantile(ref_floor, QUANTILE))
        ref_alert = ref_drift > thr_ref

        # Each month against 500 rows of the other months of its own year, same class mix.
        year_rng = np.random.default_rng(4)
        own_year = []
        for i, s in enumerate(MONTHS):
            pos = np.concatenate(
                [
                    np.arange(j * N_ROWS, (j + 1) * N_ROWS)
                    for j, m in enumerate(MONTHS)
                    if m[:4] == s[:4] and j != i
                ]
            )
            draw = np.r_[
                year_rng.choice(pos[~yA[pos]], N_ROWS - N_MAL, replace=False),
                year_rng.choice(pos[yA[pos]], N_MAL, replace=False),
            ]
            own_year.append(jaccard(tops[i], topk(absA[draw].mean(axis=0), k)))
        own_year = np.array(own_year)
        own_alert = own_year > thr_ref

        alerts = [MONTHS[i + 1] for i in np.flatnonzero(drift > thr)]
        rk = {
            "floor_p50": float(np.median(floor)),
            "floor_p95": thr,
            "floor_exceed": exceed,
            "drift": drift.tolist(),
            "drift_mean": float(drift.mean()),
            "drift_median": float(np.median(drift)),
            "alerts": alerts,
            "control_rate_mean": rate,
            "control_rate_min": float(np.min(rates)),
            "control_rate_max": float(np.max(rates)),
            "control_distance_median": float(np.median(np.concatenate(control_d))),
            "gate_pass": bool(gate),
            "background_only": bg_only,
            "background_only_mean": float(np.mean(bg_only)),
            "reference_floor_p50": float(np.median(ref_floor)),
            "reference_floor_p95": thr_ref,
            "reference_floor_exceed": float(np.mean(np.array(ref_floor) > thr_ref)),
            "reference_drift": ref_drift.tolist(),
            "reference_alerts": [MONTHS[i] for i in np.flatnonzero(ref_alert)],
            "reference_drift_mean_by_year": {
                y: float(ref_drift[[m.startswith(y) for m in MONTHS]].mean()) for y in YEARS
            },
            "reference_alerts_by_year": {
                y: int(ref_alert[[m.startswith(y) for m in MONTHS]].sum()) for y in YEARS
            },
            "own_year_drift": own_year.tolist(),
            "own_year_drift_mean_by_year": {
                y: float(own_year[[m.startswith(y) for m in MONTHS]].mean()) for y in YEARS
            },
            "own_year_alerts_by_year": {
                y: int(own_alert[[m.startswith(y) for m in MONTHS]].sum()) for y in YEARS
            },
        }
        out["k"][k] = rk
        log(
            f"  {name} k={k}: support {out['support_mean']:.0f}; month to month: drift mean {drift.mean():.3f}, "
            f"floor p50 {np.median(floor):.3f} p95 {thr:.3f} exceed {exceed:.3f}, alerts {len(alerts)}/71, "
            f"control {rate:.3f} ({np.min(rates):.3f}-{np.max(rates):.3f}); reference: floor p95 {thr_ref:.3f}, "
            f"alerts {int(ref_alert.sum())}/72; own year {own_year.mean():.3f}, above the reference floor "
            f"{int(own_alert.sum())}/72; background-only mean {np.mean(bg_only):.3f}"
        )
    out["imp"] = imp  # kept in memory for the seed spread, dropped before writing
    return out


def detection(margin: np.ndarray, yte: np.ndarray, bounds: dict, y_hold: np.ndarray) -> dict:
    """Precision, recall, F1 (percent) and flagged share per month and on the held-out rows, both rules.

    margin holds the test rows in month order, then the held-out rows.
    """
    n = len(yte)
    cut = max_f1_cutoff(y_hold, margin[n:])
    out = {}
    for rule in RULES:
        pred = margin > 0 if rule == "default" else margin >= cut
        out[rule] = {
            "cutoff": 0.0 if rule == "default" else cut,
            "month": {s: precision_recall_f1(yte[a:b], pred[a:b]) for s, (a, b) in bounds.items()},
            "flagged_month": {s: float(100 * np.mean(pred[a:b])) for s, (a, b) in bounds.items()},
            "held_out": precision_recall_f1(y_hold, pred[n:]),
            "flagged_held_out": float(100 * np.mean(pred[n:])),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    Xtr, ytr = apigraph.load_train()
    ytr = np.asarray(ytr, bool)
    parts = [apigraph.load(s) for s in MONTHS]
    Xte = np.vstack([p[0] for p in parts])
    yte = np.concatenate([p[1] for p in parts]).astype(bool)
    bounds, a = {}, 0
    for s, (_, y) in zip(MONTHS, parts, strict=True):
        bounds[s] = (a, a + len(y))
        a += len(y)
    del parts
    log(
        f"pool {Xtr.shape[0]:,} x {Xtr.shape[1]:,} ({100 * ytr.mean():.1f}% malware); test {Xte.shape[0]:,} rows "
        f"in {len(MONTHS)} months ({min(100 * yte[a:b].mean() for a, b in bounds.values()):.1f}-"
        f"{max(100 * yte[a:b].mean() for a, b in bounds.values()):.1f}% malware)"
    )

    smp = draw_sample(ytr, yte, bounds)
    digest = sample_hash(smp)
    Xfit, yfit = Xtr[smp["fit"]], ytr[smp["fit"]]
    B1, B2 = Xtr[smp["bg1"]], Xtr[smp["bg2"]]
    keysA = [f"series/{s}" for s in MONTHS] + [f"extra/{s}" for s in FLOOR_MONTHS]
    XA = np.vstack([Xte[smp[key]] for key in keysA] + [Xtr[smp["ref"]], Xtr[smp["rest"]]])
    yA = np.concatenate([yte[smp[key]] for key in keysA] + [ytr[smp["ref"]], ytr[smp["rest"]]])
    XB = np.vstack([Xte[smp[f"series/{s}"]] for s in FLOOR_MONTHS])
    Xpred = np.vstack([Xte, Xtr[smp["hold"]]])
    y_hold = ytr[smp["hold"]]
    log(
        f"fit {len(Xfit):,} rows ({100 * yfit.mean():.1f}% malware); held out {len(smp['hold']):,} "
        f"(reference {N_ROWS}, other {len(smp['rest']):,}); explained {len(XA):,} rows "
        f"({100 * yA.mean():.1f}% malware) against the deployment background, {len(XB):,} against the second"
    )
    (WORK / "phi").mkdir(parents=True, exist_ok=True)

    res: dict = {"models": {}, "time_s": {}, "sample_sha256": digest}
    imps: dict = {}

    def run(name: str, make_model, margin_fn) -> None:
        t = time.time()
        log(f"\n{name}")
        path = WORK / "phi" / f"{name}.npz"
        if args.reuse and path.exists():
            z = np.load(path)
            if str(z["sample_sha256"]) != digest:
                raise RuntimeError(f"{path} was written for a different sample")
            PA, PB, margin = z["A"], z["B"], z["margin"]
            log(f"  attributions and margins loaded from {path.relative_to(HERE)}")
        else:
            model = make_model()
            model.fit(Xfit, yfit)
            margin = np.asarray(margin_fn(model, Xpred), np.float64)
            log(f"  fit and margins {time.time() - t:.0f} s")
            PA = model.attribute(XA, B1)
            PB = model.attribute(XB, B2)
            log(f"  fit, margins, attribute {len(XA) + len(XB):,} rows in {time.time() - t:.0f} s")
            np.savez_compressed(path, A=PA, B=PB, margin=margin, sample_sha256=digest)
            for junk in ("phi.f64", "explain.tmx", "background.tmx", "score.tmx", "fit.tmx"):
                (getattr(model, "workdir", HERE / "none") / junk).unlink(missing_ok=True)
        r = analyse(name, PA, PB, yA)
        imps[name] = r.pop("imp")
        r.update(threshold_rule=make_model().threshold_rule, detection=detection(margin, yte, bounds, y_hold))
        res["models"][name] = r
        res["time_s"][name] = round(time.time() - t)

    run("logreg", LogRegDetector, chunked_margin)
    run("xgboost", lambda: XGBoostDetector(shap_jobs=SHAP_JOBS), chunked_margin)
    for seed in TM_SEEDS:
        run(
            f"fptm-seed{seed}",
            lambda seed=seed: TsetlinDetector(S=100, seed=seed, workdir=WORK / f"seed{seed}"),
            lambda m, X: m.margin(X),
        )

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
        w.writerow(["model", "k", "series", "from", "to", "distance", "threshold", "alert"])
        for name, r in res["models"].items():
            for k in KS:
                rk = r["k"][k]
                thr = rk["floor_p95"]
                for i, d in enumerate(rk["drift"]):
                    w.writerow(
                        [name, k, "month", MONTHS[i], MONTHS[i + 1], f"{d:.4f}", f"{thr:.4f}", int(d > thr)]
                    )
                thr = rk["reference_floor_p95"]
                for b, d in zip(MONTHS, rk["reference_drift"], strict=True):
                    w.writerow([name, k, "reference", "held-out", b, f"{d:.4f}", f"{thr:.4f}", int(d > thr)])

    with (HERE / "detection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            [
                "model",
                "rule",
                "cutoff",
                "period",
                "rows",
                "malware",
                "flagged_pct",
                "precision",
                "recall",
                "f1",
            ]
        )
        for name, r in res["models"].items():
            for rule in RULES:
                d = r["detection"][rule]
                periods = [
                    (s, yte[a:b], d["month"][s], d["flagged_month"][s]) for s, (a, b) in bounds.items()
                ]
                periods.append(("held-out", y_hold, d["held_out"], d["flagged_held_out"]))
                for period, y, (p, rc, f), fl in periods:
                    w.writerow(
                        [name, rule, f"{d['cutoff']:.6g}", period, len(y), int(y.sum()), f"{fl:.2f}"]
                        + [f"{p:.2f}", f"{rc:.2f}", f"{f:.2f}"]
                    )

    log(
        f"\n=== gate at k={GATE_K}: shuffled control alert rate between e/2 and 2e (e = floor exceedance) ==="
    )
    ok = True
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        e = rk["floor_exceed"]
        ok &= rk["gate_pass"]
        log(
            f"{name:<13} e {e:.3f}  control {rk['control_rate_mean']:.3f} in [{e / CONTROL_FACTOR:.3f}, "
            f"{CONTROL_FACTOR * e:.3f}] {'yes' if rk['gate_pass'] else 'NO '}  alerts {len(rk['alerts']):>2}/71  "
            f"floor p50 {rk['floor_p50']:.3f} p95 {rk['floor_p95']:.3f}  drift median {rk['drift_median']:.3f}  "
            f"control distance median {rk['control_distance_median']:.3f}"
        )
    log(
        f"\n=== month-to-month alerts at k={GATE_K} (month: distance / floor p95, F1 at the held-out cut-off, "
        "previous -> this month) ==="
    )
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        fm = r["detection"]["held-out"]["month"]
        items = [
            f"{s} {rk['drift'][MONTHS.index(s) - 1]:.3f}/{rk['floor_p95']:.3f} "
            f"({fm[MONTHS[MONTHS.index(s) - 1]][2]:.1f}->{fm[s][2]:.1f})"
            for s in rk["alerts"]
        ]
        log(f"{name}: " + ("; ".join(items) if items else "none"))
    log(f"\n=== distance from the reference at k={GATE_K}, mean by year; alerts by year (of 12) ===")
    log(f"{'model':<13} ref floor p95  " + "  ".join(f"{y:>11}" for y in YEARS))
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        cells = "  ".join(
            f"{rk['reference_drift_mean_by_year'][y]:.3f} ({rk['reference_alerts_by_year'][y]:>2})"
            for y in YEARS
        )
        log(f"{name:<13} {rk['reference_floor_p95']:>13.3f}  {cells}")
    log(
        f"\n=== distance from the other months of the same year at k={GATE_K}, mean by year; months above the "
        "reference floor p95 (of 12) ==="
    )
    log(f"{'model':<13} ref floor p95  " + "  ".join(f"{y:>11}" for y in YEARS))
    for name, r in res["models"].items():
        rk = r["k"][GATE_K]
        cells = "  ".join(
            f"{rk['own_year_drift_mean_by_year'][y]:.3f} ({rk['own_year_alerts_by_year'][y]:>2})"
            for y in YEARS
        )
        log(f"{name:<13} {rk['reference_floor_p95']:>13.3f}  {cells}")
    log(
        "\n=== detection: precision / recall / F1 (percent); held-out 2012 rows, then mean of 12 months per year ==="
    )
    for name, r in res["models"].items():
        for rule in RULES:
            d = r["detection"][rule]
            h = d["held_out"]
            cells = "  ".join(
                f"{y} "
                + "/".join(
                    f"{v:.1f}" for v in np.mean([d["month"][m] for m in MONTHS if m.startswith(y)], axis=0)
                )
                for y in YEARS
            )
            log(
                f"{name:<13} {rule:<8} cut-off {d['cutoff']:>9.4g}  held-out {h[0]:.1f}/{h[1]:.1f}/{h[2]:.1f}  {cells}"
            )

    verdict = "PASS" if ok else "FAIL"
    res.update(
        verdict=verdict,
        config=dict(
            months=MONTHS,
            floor_months=FLOOR_MONTHS,
            n_rows=N_ROWS,
            n_mal=N_MAL,
            n_bg=N_BG,
            n_bg_mal=N_BG_MAL,
            holdout=HOLDOUT,
            n_fit=len(smp["fit"]),
            n_hold=len(smp["hold"]),
            n_splits=N_SPLITS,
            n_ref_draws=N_REF_DRAWS,
            n_shuffles=N_SHUFFLES,
            ks=KS,
            gate_k=GATE_K,
            quantile=QUANTILE,
            control_factor=CONTROL_FACTOR,
            tm_seeds=TM_SEEDS,
            shap_jobs=SHAP_JOBS,
            rules=RULES,
        ),
        platform=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    )
    res["time_s"]["total"] = round(time.time() - t0)
    log(f"\nRESULT apigraph_drift_reference={verdict}  ({res['time_s']['total']} s)")
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
