"""The replay's steps, independent of Dagster: set-up on the 2012 pool, then per month score, detect,
drift, alert, learn and summarise. Every step reads its inputs from disk and writes its results to the
state database, so each can run in its own process.

Labels of a month are taken as known at the end of that month: detection, drift, alerts and learning
use them, scoring does not.

Each (model, arm) has numbered model versions. Version 0 is the 2012 model, shared by every arm. A
version directory holds the model, the background its attributions are taken against and its drift
reference (`reference.npz`: importance and top-k set of the reference rows, and the reference floor).
"""

from __future__ import annotations

import json
import pickle
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from tmcyberops import tmx
from tmcyberops.datasets import apigraph
from tmcyberops.datasets.drebin_labels import label as feature_label
from tmcyberops.drift import importance, jaccard, pick, ranks, topk
from tmcyberops.models import (
    LogRegDetector,
    TsetlinDetector,
    XGBoostDetector,
    max_f1_cutoff,
    precision_recall_f1,
)
from tmcyberops.platform import config, llm, state
from tmcyberops.platform.config import ARM_LABELS, MODEL_LABELS

N_FEATURES_SHOWN = 10  # top features stored per month
N_CHANGES_SHOWN = 5  # features entering / leaving the top-k stored per month


# ---------------------------------------------------------------- models on disk


def model_dir(model: str, arm: str, version: int) -> Path:
    name = "base" if version == 0 else f"{arm}-v{version}"
    return config.home() / "models" / model / name


def month_dir(month: str) -> Path:
    p = config.home() / "months" / month
    p.mkdir(parents=True, exist_ok=True)
    return p


def month_index(month: str) -> int:
    y, m = map(int, month.split("-"))
    return 12 * y + m - 1


def make_detector(model: str, cfg: dict, d: Path):
    if model == "fptm":
        return TsetlinDetector(seed=cfg["fptm_seed"], workdir=d)
    if model == "xgboost":
        return XGBoostDetector(shap_jobs=cfg["xgboost_shap_jobs"])
    if model == "logreg":
        return LogRegDetector()
    raise ValueError(f"unknown model {model!r}")


def save_detector(model: str, det, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if model == "xgboost":
        det.clf.save_model(d / "model.json")
    elif model == "logreg":
        (d / "model.pkl").write_bytes(pickle.dumps(det.clf))
    # fptm: the fitted .tmcore file already lives in d


def load_detector(model: str, cfg: dict, d: Path):
    det = make_detector(model, cfg, d)
    if model == "xgboost":
        import xgboost as xgb

        det.clf = xgb.XGBClassifier()
        det.clf.load_model(d / "model.json")
    elif model == "logreg":
        det.clf = pickle.loads((d / "model.pkl").read_bytes())  # written by save_detector above
    return det


def current(model: str, arm: str) -> dict:
    """The model version an arm uses now, with its cut-off and reference floor."""
    df = state.query(
        "select * from models where model = ? and arm = ? and status = 'active' order by version desc limit 1",
        [model, arm],
    )
    if not len(df):
        raise RuntimeError(f"no model for {model}/{arm}: run the set-up (asset models_2012) first")
    rec = df.iloc[0].to_dict()
    rec["dir"] = model_dir(model, arm, int(rec["version"]))
    return rec


def next_version(model: str, arm: str) -> int:
    df = state.query("select max(version) as v from models where model = ? and arm = ?", [model, arm])
    return int(df["v"][0]) + 1


def labels(month: str) -> np.ndarray:
    meta = tmx.read_arrow(apigraph.tmx_dir() / f"{month}.meta.arrow")
    return meta.column("label").to_numpy(zero_copy_only=False).astype(bool)


def who(model: str, arm: str, version: int | None = None) -> str:
    v = "" if version is None else f", v{version}"
    return f"{MODEL_LABELS[model]} ({ARM_LABELS[arm]}{v})"


# ---------------------------------------------------------------- shared pieces


def stratified_holdout(rng: np.random.Generator, y: np.ndarray, share: float) -> np.ndarray:
    return np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(y == c), round(share * np.sum(y == c)), replace=False)
                for c in (False, True)
            ]
        )
    )


def heldout_stats(det, X: np.ndarray, y: np.ndarray) -> dict:
    """Max-F1 cut-off on held-out rows, with precision / recall / F1 at the default rule and F1 at the cut-off."""
    m = det.margin(X)
    cut = max_f1_cutoff(y, m)
    p, r, f = precision_recall_f1(y, m > 0)
    return dict(
        cutoff=cut,
        heldout_precision=p,
        heldout_recall=r,
        heldout_f1=f,
        heldout_f1_cutoff=precision_recall_f1(y, m >= cut)[2],
    )


def build_reference(cfg: dict, det, X_ref, X_other, y_other, bg, seed: int) -> dict:
    """Importance and top-k set of the reference rows, and the reference floor: the distance of
    `reference_floor_draws` stratified draws of other held-out rows from that set."""
    s, k = cfg["sample"], cfg["drift"]["top_k"]
    n, n_mal = s["explain_rows"], s["explain_malware"]
    phi = det.attribute(np.vstack([X_ref, X_other]), bg)
    imp = importance(phi[: len(X_ref)])
    top = topk(imp, k)
    phi_other = phi[len(X_ref) :]
    rng = np.random.default_rng(seed)
    floor = np.array(
        [
            jaccard(
                topk(importance(phi_other[pick(rng, np.arange(len(y_other)), y_other, n - n_mal, n_mal)]), k),
                top,
            )
            for _ in range(s["reference_floor_draws"])
        ]
    )
    return dict(imp=imp, top=top, floor=floor)


def register(
    cfg: dict, model: str, arm: str, version: int, trained_on: str, rows: int, stats: dict, floor, **kw
):
    row = dict(
        model=model,
        arm=arm,
        version=version,
        trained_on=trained_on,
        rows=rows,
        **stats,
        reference_floor_p95=float(np.quantile(floor, cfg["drift"]["quantile"])),
        status=kw.get("status", "active"),
        proposed_month=kw.get("proposed_month"),
        decided_by=kw.get("decided_by"),
        decided_ts=state.now() if kw.get("decided_by") else None,
        ts=state.now(),
    )
    state.upsert("models", [row])
    return row


# ---------------------------------------------------------------- set-up on the 2012 pool


def bootstrap(cfg: dict) -> list[dict]:
    """Train every model on the 2012 pool, choose its held-out cut-off and build its drift reference."""
    s = cfg["sample"]
    X, y = apigraph.load_train()
    rng = np.random.default_rng(0)
    hold = stratified_holdout(rng, y, s["holdout_share"])
    fit = np.setdiff1d(np.arange(len(y)), hold)
    bg = pick(rng, fit, y, s["background_rows"] - s["background_malware"], s["background_malware"])
    ref = pick(rng, hold, y, s["explain_rows"] - s["explain_malware"], s["explain_malware"])
    other = np.setdiff1d(hold, ref)
    # held-out 2012 rows, kept for the incremental arm's cut-off and reference
    np.savez(config.home() / "base_holdout.npz", X=X[hold], y=y[hold], ref=np.searchsorted(hold, ref))
    rows = []
    for model in cfg["models"]:
        d = model_dir(model, "base", 0)
        det = make_detector(model, cfg, d).fit(X[fit], y[fit])
        save_detector(model, det, d)
        np.save(d / "background.npy", X[bg])
        stats = heldout_stats(det, X[hold], y[hold])
        ref_data = build_reference(cfg, det, X[ref], X[other], y[other], X[bg], seed=3)
        np.savez(d / "reference.npz", **ref_data)
        for arm in cfg["arms"]:
            rows.append(
                register(
                    cfg, model, arm, 0, "2012 pool", len(fit), stats, ref_data["floor"], decided_by="set-up"
                )
            )
    return rows


# ---------------------------------------------------------------- monthly steps


def resolve_pending(cfg: dict, month: str) -> list[dict]:
    """Promote or drop retrain candidates: by the decision recorded from n8n, or by the timeout."""
    timeout = float(cfg["retrain"]["approval_timeout_minutes"])
    events = []
    for _, r in state.query("select * from models where status = 'pending'").iterrows():
        key = [r["model"], r["arm"], int(r["version"])]
        dec = state.query(
            "select decision, decided_by from decisions where model = ? and arm = ? and version = ? "
            "order by ts desc limit 1",
            key,
        )
        if len(dec):
            decision, by = dec["decision"][0], dec["decided_by"][0]
        elif (state.now() - r["ts"]).total_seconds() > 60 * timeout:
            decision, by = "approve", f"timeout ({timeout:g} min)"
        else:
            continue
        name = who(r["model"], r["arm"], int(r["version"]))
        if decision == "approve":
            state.execute(
                "update models set status = 'retired' where model = ? and arm = ? and status = 'active'",
                key[:2],
            )
            state.execute(
                "update models set status = 'active', decided_by = ?, decided_ts = ? "
                "where model = ? and arm = ? and version = ?",
                [by, state.now(), *key],
            )
            state.execute("update alert_state set active = false where model = ? and arm = ?", key[:2])
            msg = f"{name} promoted ({by}); trained on {r['trained_on']}"
            events.append(event_row(month, r["model"], r["arm"], "retrain promoted", msg))
        else:
            state.execute(
                "update models set status = 'rejected', decided_by = ?, decided_ts = ? "
                "where model = ? and arm = ? and version = ?",
                [by, state.now(), *key],
            )
            events.append(
                event_row(month, r["model"], r["arm"], "retrain rejected", f"{name} rejected ({by})")
            )
    state.append("events", events)
    return events


def record_decision(model: str, arm: str, version: int, decision: str, decided_by: str) -> None:
    """Called by the Dagster job `decision`, which n8n launches when someone answers the approval form."""
    if decision not in ("approve", "reject"):
        raise ValueError(f"decision must be approve or reject, not {decision!r}")
    state.append(
        "decisions",
        [
            dict(
                ts=state.now(),
                model=model,
                arm=arm,
                version=version,
                decision=decision,
                decided_by=decided_by,
            )
        ],
    )


def step_scores(cfg: dict, month: str) -> dict:
    out = {"decisions applied": len(resolve_pending(cfg, month))}
    X = apigraph.load(month)[0]
    md = month_dir(month)
    for model, arm in config.pairs(cfg):
        rec = current(model, arm)
        m = load_detector(model, cfg, rec["dir"]).margin(X)
        np.save(md / f"{model}-{arm}.margin.npy", m)
        out[f"{model}/{arm} flagged %"] = round(100 * float(np.mean(m > 0)), 2)
    out["rows"] = len(X)
    return out


def step_detection(cfg: dict, month: str) -> dict:
    y = labels(month)
    md = month_dir(month)
    rows, out = [], {}
    for model, arm in config.pairs(cfg):
        rec = current(model, arm)
        m = np.load(md / f"{model}-{arm}.margin.npy")
        for rule, cut, pred in (("default", 0.0, m > 0), ("held-out", rec["cutoff"], m >= rec["cutoff"])):
            p, r, f = precision_recall_f1(y, pred)
            rows.append(
                dict(
                    month=month,
                    model=model,
                    arm=arm,
                    version=int(rec["version"]),
                    rule=rule,
                    cutoff=float(cut),
                    rows=len(y),
                    malware=int(y.sum()),
                    flagged_pct=100 * float(pred.mean()),
                    precision=p,
                    recall=r,
                    f1=f,
                )
            )
            out[f"{model}/{arm} F1 {rule}"] = round(f, 1)
    state.upsert("metrics", rows)
    return out


def step_drift(cfg: dict, month: str) -> dict:
    s, k = cfg["sample"], cfg["drift"]["top_k"]
    X, y = apigraph.load(month)
    rng = np.random.default_rng(int(month.replace("-", "")))
    rows_idx = pick(rng, np.arange(len(y)), y, s["explain_rows"] - s["explain_malware"], s["explain_malware"])
    names = apigraph.features()
    md, pmd = month_dir(month), config.home() / "months" / config.previous_month(month)
    drift_rows, feat_rows, out = [], [], {}
    for model, arm in config.pairs(cfg):
        rec = current(model, arm)
        det = load_detector(model, cfg, rec["dir"])
        imp = importance(det.attribute(X[rows_idx], np.load(rec["dir"] / "background.npy")))
        np.save(md / f"{model}-{arm}.importance.npy", imp)
        top = topk(imp, k)
        ref = np.load(rec["dir"] / "reference.npz")
        d_ref = jaccard(top, ref["top"])
        p95 = float(rec["reference_floor_p95"])
        base = dict(month=month, model=model, arm=arm, version=int(rec["version"]), k=k)
        drift_rows.append(base | dict(versus="reference", distance=d_ref, floor_p95=p95, above=d_ref > p95))
        out[f"{model}/{arm} distance from reference"] = round(d_ref, 3)
        prev = pmd / f"{model}-{arm}.importance.npy"
        if prev.exists():
            d_prev = jaccard(top, topk(np.load(prev), k))
            drift_rows.append(
                base | dict(versus="previous month", distance=d_prev, floor_p95=None, above=None)
            )
        r_now, r_ref = ranks(imp), ranks(ref["imp"])
        entered = [j for j in np.argsort(-imp, kind="stable") if top[j] and not ref["top"][j]]
        left = [j for j in np.argsort(-ref["imp"], kind="stable") if ref["top"][j] and not top[j]]
        groups = (
            ("top", np.argsort(-imp, kind="stable")[:N_FEATURES_SHOWN]),
            ("entered", entered[:N_CHANGES_SHOWN]),
            ("left", left[:N_CHANGES_SHOWN]),
        )
        for kind, idx in groups:
            for pos, j in enumerate(idx, 1):
                feat_rows.append(
                    dict(
                        month=month,
                        model=model,
                        arm=arm,
                        kind=kind,
                        position=pos,
                        feature=names[j],
                        importance=float(imp[j]),
                        rank=int(r_now[j]),
                        reference_rank=int(r_ref[j]),
                    )
                )
    state.upsert("drift", drift_rows)
    state.upsert("features", feat_rows)
    return out


def step_alert(cfg: dict, month: str) -> dict:
    """Raise or clear alerts on state changes and send raised ones to the n8n webhook."""
    a = cfg["alerts"]
    events, states = [], []
    for model, arm in config.pairs(cfg):
        rec = current(model, arm)
        name = who(model, arm, int(rec["version"]))
        # a retrained version starts a new reference, so its months count from its promotion; the
        # incremental arm changes version every month but keeps the 2012 reference rows and floor
        min_version = int(rec["version"]) if arm == "retrain" else 0
        d = state.query(
            "select month, above, distance, floor_p95 from drift where model = ? and arm = ? and version >= ? "
            "and versus = 'reference' and month <= ? order by month desc limit ?",
            [model, arm, min_version, month, a["reference_drift_months"]],
        )
        drift_firing = len(d) == a["reference_drift_months"] and bool(d["above"].all())
        m = state.query(
            "select f1 from metrics where month = ? and model = ? and arm = ? and rule = 'held-out'",
            [month, model, arm],
        )
        line = float(rec["heldout_f1_cutoff"]) - a["f1_drop_points"]
        f1_now = float(m["f1"][0]) if len(m) else float("nan")
        checks = (
            (
                "drift",
                drift_firing,
                f"{name}: attribution set {a['reference_drift_months']} months above its reference floor "
                f"(distance {d['distance'].iloc[0]:.3f} > {d['floor_p95'].iloc[0]:.3f})"
                if len(d)
                else "",
            ),
            (
                "f1",
                bool(len(m)) and f1_now < line,
                f"{name}: F1 {f1_now:.1f} at the held-out cut-off, below {line:.1f}",
            ),
        )
        prev = state.query("select kind, active from alert_state where model = ? and arm = ?", [model, arm])
        active = dict(zip(prev["kind"], prev["active"], strict=True)) if len(prev) else {}
        for kind, firing, msg in checks:
            was = bool(active.get(kind, False))
            if firing and not was:
                events.append(send(cfg, event_row(month, model, arm, kind, msg)))
            elif was and not firing:
                events.append(
                    event_row(month, model, arm, f"{kind} cleared", f"{name}: {kind} alert cleared")
                )
            states.append(dict(model=model, arm=arm, kind=kind, active=firing))
    state.append("events", events)
    state.upsert("alert_state", states)
    return {"events": len(events)}


def step_learn(cfg: dict, month: str) -> dict:
    """Update the learning arms with the labels of `month`. The frozen arm never changes."""
    out = {}
    if "incremental" in cfg["arms"]:
        X, y = apigraph.load(month)
        for model in cfg["models"]:
            out[f"{model}/incremental"] = learn_incremental(cfg, model, month, X, y)
    if "retrain" in cfg["arms"]:
        for model in cfg["models"]:
            out[f"{model}/retrain"] = learn_retrain(cfg, model, month)
    return out


def learn_incremental(cfg: dict, model: str, month: str, X: np.ndarray, y: np.ndarray) -> str:
    """The current model learns the month's rows. Its cut-off is chosen again on the held-out 2012 rows
    and its reference rows (500 held-out 2012 rows) are explained again; the floor stays the 2012 one."""
    inc = cfg["incremental"]
    rec = current(model, "incremental")
    det = load_detector(model, cfg, rec["dir"])
    v = next_version(model, "incremental")
    d = model_dir(model, "incremental", v)
    d.mkdir(parents=True, exist_ok=True)
    if model == "fptm":
        new = det.update(X, y, epochs=inc["fptm_epochs"], workdir=d, rng_offset=100 * month_index(month))
    elif model == "xgboost":
        new = det.update(X, y, trees=inc["xgboost_trees"])
    else:
        new = det.update(X, y)
    save_detector(model, new, d)
    base = model_dir(model, "base", 0)
    shutil.copy(base / "background.npy", d / "background.npy")
    H = np.load(config.home() / "base_holdout.npz")
    imp = importance(new.attribute(H["X"][H["ref"]], np.load(d / "background.npy")))
    floor = np.load(base / "reference.npz")["floor"]
    np.savez(d / "reference.npz", imp=imp, top=topk(imp, cfg["drift"]["top_k"]), floor=floor)
    register(
        cfg,
        model,
        "incremental",
        v,
        f"2012 pool, then each month up to {month}",
        len(y),
        heldout_stats(new, H["X"], H["y"]),
        floor,
        decided_by="incremental",
    )
    return f"v{v}"


def learn_retrain(cfg: dict, model: str, month: str) -> str:
    """While the retrain arm's alert is active, train a candidate on the recent labelled months and ask
    for approval; at most one candidate waits at a time."""
    rc, s = cfg["retrain"], cfg["sample"]
    rows = state.query("select * from models where model = ? and arm = 'retrain'", [model])
    if (rows["status"] == "pending").any():
        return "candidate awaiting approval"
    if not len(
        state.query("select 1 from alert_state where model = ? and arm = 'retrain' and active", [model])
    ):
        return "no alert"
    proposed = rows["proposed_month"].dropna()
    if len(proposed) and month_index(month) - month_index(max(proposed)) < rc["repropose_months"]:
        return "proposed recently"

    window = [m for m in config.months(cfg) if m <= month][-rc["window_months"] :]
    parts = [apigraph.load(m) for m in window]
    trained_on = f"{window[0]}..{window[-1]}"
    if len(window) < rc["window_months"]:
        parts.insert(0, apigraph.load_train())
        trained_on = f"2012 pool + {trained_on}"
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    rng = np.random.default_rng(month_index(month))
    hold = stratified_holdout(rng, y, s["holdout_share"])
    fit = np.setdiff1d(np.arange(len(y)), hold)
    bg = pick(rng, fit, y, s["background_rows"] - s["background_malware"], s["background_malware"])
    ref = pick(rng, hold, y, s["explain_rows"] - s["explain_malware"], s["explain_malware"])
    other = np.setdiff1d(hold, ref)

    v = next_version(model, "retrain")
    d = model_dir(model, "retrain", v)
    det = make_detector(model, cfg, d).fit(X[fit], y[fit])
    save_detector(model, det, d)
    np.save(d / "background.npy", X[bg])
    stats = heldout_stats(det, X[hold], y[hold])
    ref_data = build_reference(cfg, det, X[ref], X[other], y[other], X[bg], seed=month_index(month))
    np.savez(d / "reference.npz", **ref_data)
    register(
        cfg,
        model,
        "retrain",
        v,
        trained_on,
        len(fit),
        stats,
        ref_data["floor"],
        status="pending",
        proposed_month=month,
    )
    msg = (
        f"{who(model, 'retrain', v)} trained on {trained_on} ({len(fit):,} rows): held-out F1 "
        f"{stats['heldout_f1_cutoff']:.1f}. Approve or reject; it replaces the current model by itself after "
        f"{rc['approval_timeout_minutes']} min."
    )
    ev = send(cfg, event_row(month, model, "retrain", "retrain proposed", msg), version=v)
    state.append("events", [ev])
    return f"candidate v{v} proposed"


def step_summary(cfg: dict, month: str) -> dict:
    """A short text per model and arm, filled from the month's numbers, and one summary of the month by
    each configured language model (ops/platform.yaml `llm`). Last step of a month: marks it done."""
    rows = [
        dict(
            month=month,
            model=model,
            arm=arm,
            author="template",
            text=template_summary(month, model, arm),
            ts=state.now(),
        )
        for model, arm in config.pairs(cfg)
    ]
    facts = "\n".join(r["text"] for r in rows)
    for name, draft in llm.summarise(cfg, month, facts).items():
        text = draft["text"] if not draft["rejected"] else f"[not shown: {draft['rejected']}] {draft['text']}"
        rows.append(dict(month=month, model="all", arm="all", author=name, text=text, ts=state.now()))
    state.upsert("summaries", rows)
    state.upsert("months", [dict(month=month, ts=state.now())])
    return {"summaries": len(rows)}


def template_summary(month: str, model: str, arm: str) -> str:
    m = state.query("select * from metrics where month = ? and model = ? and arm = ?", [month, model, arm])
    d = state.query("select * from drift where month = ? and model = ? and arm = ?", [month, model, arm])
    f = state.query(
        "select * from features where month = ? and model = ? and arm = ? order by kind, position",
        [month, model, arm],
    )
    e = state.query(
        "select kind, message from events where month = ? and model = ? and arm = ? order by ts",
        [month, model, arm],
    )
    version = int(m["version"].iloc[0]) if len(m) else None
    parts = [f"{month}, {who(model, arm, version)}:"]
    by_rule = {r["rule"]: r for _, r in m.iterrows()}
    if "held-out" in by_rule:
        h, dft = by_rule["held-out"], by_rule["default"]
        parts.append(
            f"F1 {h['f1']:.1f} at the held-out cut-off (precision {h['precision']:.1f}, recall "
            f"{h['recall']:.1f}); {dft['f1']:.1f} at the default rule."
        )
    ref = d[d["versus"] == "reference"]
    if len(ref):
        r = ref.iloc[0]
        side = "above" if r["above"] else "within"
        parts.append(
            f"Top-{int(r['k'])} attribution set vs the model's reference rows: distance {r['distance']:.3f}, "
            f"{side} its floor ({r['floor_p95']:.3f})."
        )
    prev = d[d["versus"] == "previous month"]
    if len(prev):
        parts.append(f"Distance from the previous month: {prev.iloc[0]['distance']:.3f}.")

    def words(feature: str) -> str:
        category, text = feature_label(feature)
        return f"{text} ({category.lower()})"

    top = [words(x) for x in f[f["kind"] == "top"]["feature"].tolist()[:3]]
    if top:
        parts.append(f"{'; '.join(top)} contributed most to the model's malware score.")
    ent = [words(x) for x in f[f["kind"] == "entered"]["feature"].tolist()[:3]]
    if ent and len(ref):
        parts.append(f"New in the top {int(ref.iloc[0]['k'])}: {'; '.join(ent)}.")
    for _, ev in e.iterrows():
        parts.append(f"Event: {ev['kind']}.")
    return " ".join(parts)


# ---------------------------------------------------------------- events and n8n


def event_row(month: str, model: str, arm: str, kind: str, message: str) -> dict:
    return dict(
        ts=state.now(), month=month, model=model, arm=arm, kind=kind, message=message, delivery="", link=None
    )


def send(cfg: dict, ev: dict, **extra) -> dict:
    """POST the event to the n8n webhook; n8n may answer with the link of its approval form."""
    svc = cfg["services"]
    payload = {k: ev[k] for k in ("month", "model", "arm", "kind", "message")} | extra
    payload |= dict(
        replay=state.replay_id(),
        dashboard=f"http://localhost:{svc['dashboard_port']}",
        dagster=f"http://localhost:{svc['dagster_port']}",
    )
    try:
        req = urllib.request.Request(
            svc["n8n_webhook"],
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # local n8n only
            ev["delivery"] = f"n8n {resp.status}"
            body = resp.read().decode("utf-8", "replace")
        try:
            ev["link"] = json.loads(body).get("form")
        except (ValueError, AttributeError):
            pass
    except Exception as e:  # n8n down or its workflow inactive: the event still lands in the state
        ev["delivery"] = f"n8n not reached ({type(e).__name__})"
    return ev
