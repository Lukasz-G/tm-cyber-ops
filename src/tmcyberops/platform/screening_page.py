"""Package screening page: which lines of a package made each detector call it malicious or benign.

    uv run streamlit run src/tmcyberops/platform/screening_page.py --server.port 8502

Reads the output of `python -m tmcyberops.platform.screening build`. Source code is read from the
archives in memory each time a package is shown, never unpacked or run; URLs, IP addresses, e-mail
addresses and long encoded strings are masked before display. `?package=<id>` opens one package.
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tmcyberops.datasets import codefeatures, datadog
from tmcyberops.platform.screening import LABELS, MODELS, out_dir

SHOWN = ("fptm", "xgboost")  # side by side; logistic regression appears in the verdict row
MAX_LINE = 150
WINDOW = 34
# the replay dashboard's palette (dashboard.py): dark navy theme, the frozen models' blue, muted text
FROZEN = {"light": "#2a78d6", "dark": "#3987e5"}
MUTED = {"light": "#52514e", "dark": "#9aa7bd"}
GRID = {"light": "rgba(11,11,11,0.07)", "dark": "rgba(154,167,189,0.13)"}
TOWARDS_MALICIOUS, TOWARDS_BENIGN = (229, 72, 77), (57, 135, 229)

st.set_page_config(page_title="Package screening", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    .tile {padding:0.7rem 0.9rem; border-radius:10px; background:rgba(127,156,199,0.10)}
    .tile .t {font-size:0.82rem; opacity:0.75}
    .tile .v {font-size:1.8rem; font-weight:600; line-height:1.25}
    .tile .s {font-size:0.82rem; opacity:0.75}
    .code {font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; line-height: 1.45;
           border-radius: 10px; background: rgba(127,156,199,0.06); overflow-x: auto; padding: 6px 0;}
    .code .l {white-space: pre; padding: 0 10px;}
    .code .n {display: inline-block; width: 3.2em; opacity: 0.5; user-select: none;}
    .verdict {padding:0.55rem 0.9rem; border-radius:10px; background:rgba(127,156,199,0.10)}
    .verdict small {opacity:0.75}
    .mal {color: #ef6b6f; font-weight: 600;} .ben {color: #5598e7; font-weight: 600;}
    </style>""",
    unsafe_allow_html=True,
)


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tile(col, title: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="tile"><div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load():
    d = out_dir()
    rows = json.loads((d / "packages.json").read_text(encoding="utf-8"))
    metrics = pd.DataFrame(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    explained = json.loads((d / "explained.json").read_text(encoding="utf-8"))
    feats = json.loads((d / "features.json").read_text(encoding="utf-8"))
    phi = {m: np.load(d / f"phi-{m}.npy") for m in MODELS}
    return rows, metrics, meta, explained, feats, phi


@st.cache_data
def occurrence_index() -> dict[str, int]:
    return codefeatures.occurrence_index()


@st.cache_data(max_entries=256)
def occurrence(pid: str) -> dict:
    return codefeatures.occurrence(pid, occurrence_index())


@st.cache_data(max_entries=64)
def source(pid: str, row_json: str) -> dict[str, str]:
    files = datadog.python_files(json.loads(row_json))
    return {k: files[k] for k in sorted(files)}


def verdict(row: dict, model: str) -> tuple[bool, bool]:
    called = row[f"{model}_cutoff"]
    return called, called == bool(row["label"])


def word(mal: bool) -> str:
    return '<span class="mal">malicious</span>' if mal else '<span class="ben">benign</span>'


def code_html(text: str, heat: dict, top: float, lines: range) -> str:
    out = []
    src = text.splitlines()
    for ln in lines:
        if ln < 1 or ln > len(src):
            continue
        t = codefeatures.mask(src[ln - 1])
        t = t if len(t) <= MAX_LINE else t[:MAX_LINE] + " …"
        h = heat.get(ln, 0.0)
        a = min(0.85, abs(h) / top) if top > 0 else 0.0
        r, g, b = TOWARDS_MALICIOUS if h > 0 else TOWARDS_BENIGN
        bg = f"rgba({r},{g},{b},{a:.2f})"
        out.append(
            f'<div class="l" style="background:{bg}"><span class="n">{ln}</span>{html.escape(t)}</div>'
        )
    return '<div class="code">' + "".join(out) + "</div>"


rows, metrics, meta, explained, feats, phi = load()
by_id = {r["id"]: r for r in rows}
row_of = {pid: i for i, pid in enumerate(explained)}
test = [by_id[p] for p in explained]
quarters = sorted({r["quarter"] for r in test})

st.title("Package screening")
st.caption(
    "Malicious PyPI packages from Datadog's dataset and random PyPI packages from the same quarters, "
    f"matched in size. FPTM, XGBoost and logistic regression are trained on packages published before "
    f"{meta['train_end'][:7]} and score every later package. Source is read from the archives in memory and "
    "never run; URLs, IP addresses, e-mail addresses and encoded strings are masked."
)
n_mal = sum(r["label"] for r in test)
c = st.columns(4)
tile(
    c[0], "Training packages", f"{meta['fit'] + meta['held']:,}", "published before " + meta["train_end"][:7]
)
tile(c[1], "Screened after", f"{len(test):,}", f"{n_mal:,} malicious, {len(test) - n_mal:,} benign")
tile(c[2], "Quarters", str(len(quarters)), f"{quarters[0]} to {quarters[-1]}")
tile(c[3], "Features", f"{len(feats):,}", "imports, calls, names, string classes")
st.write("")

tab_screen, tab_quarters = st.tabs(["Screen a package", "Detection per quarter"])

with tab_quarters:
    from plotly.subplots import make_subplots

    t = theme()
    m = metrics[(metrics["rule"] == "cutoff") & (metrics["period"].isin(quarters))]
    fig = make_subplots(rows=1, cols=len(MODELS), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[LABELS[x] for x in MODELS])  # fmt: skip
    for k, model in enumerate(MODELS, 1):
        d = m[m["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=d["period"], y=d["f1"], mode="lines+markers", line=dict(color=FROZEN[t], width=2.2),
                marker=dict(size=6), customdata=np.c_[d["malicious"], d["benign"], d["precision"], d["recall"]],
                hovertemplate="F1 %{y:.1f} (P %{customdata[2]:.1f}, R %{customdata[3]:.1f}; "
                "%{customdata[0]} malicious, %{customdata[1]} benign)<extra></extra>",
            ),
            row=1, col=k,
        )  # fmt: skip
    fig.update_layout(height=300, margin=dict(l=4, r=4, t=28, b=4), hovermode="x unified", showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")  # fmt: skip
    fig.update_yaxes(gridcolor=GRID[t], zeroline=False, range=[0, 101])
    fig.update_yaxes(title_text="F1 (%)", row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10, color=MUTED[t]))
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", theme="streamlit", key="f1q")
    st.caption(
        "F1 of the malicious class per quarter of publication, frozen detectors trained before "
        f"{meta['train_end'][:7]}, at each detector's max-F1 cut-off chosen on 10% of the training packages "
        "held out. Quarters differ in size (hover for counts); 2023Q3 to 2024Q1 hold few malicious packages."
    )
    st.dataframe(
        m.pivot_table(index="period", columns="model", values="f1")[list(MODELS)]
        .rename(columns=LABELS)
        .round(1),
        width="stretch",
    )

with tab_screen:
    want = st.query_params.get("package")
    left, mid, right = st.columns([1, 1, 2])
    default_q = by_id[want]["quarter"] if want in by_id else quarters[-1]
    q = left.selectbox("Quarter", quarters, index=quarters.index(default_q))
    kind = mid.selectbox(
        "Show", ["all", "malicious", "benign", "the models disagree", "missed or false alarm"]
    )
    pool = [r for r in test if r["quarter"] == q]
    if kind == "malicious":
        pool = [r for r in pool if r["label"]]
    elif kind == "benign":
        pool = [r for r in pool if not r["label"]]
    elif kind == "the models disagree":
        pool = [r for r in pool if len({r[f"{m}_cutoff"] for m in MODELS}) > 1]
    elif kind == "missed or false alarm":
        pool = [r for r in pool if any(not verdict(r, m)[1] for m in SHOWN)]
    if not pool:
        st.info("No package of this kind in this quarter.")
        st.stop()

    def label(r: dict) -> str:
        calls = " ".join(f"{LABELS[m].split()[0]} {'✓' if verdict(r, m)[1] else '✗'}" for m in SHOWN)
        return f"{r['name']} {r['version']}  ·  {'malicious' if r['label'] else 'benign'}  ·  {calls}"

    ids = [r["id"] for r in pool]
    pick = right.selectbox("Package", ids, index=ids.index(want) if want in ids else 0,
                           format_func=lambda i: label(by_id[i]))  # fmt: skip
    row = by_id[pick]
    i = row_of[pick]
    occ = occurrence(pick)
    files = source(pick, json.dumps(row))
    names = occ["files"]

    truth = "malicious (Datadog)" if row["label"] else "benign (random PyPI package)"
    st.markdown(
        f"**{html.escape(row['name'])} {html.escape(row['version'])}**, published {row['date']}, "
        f"{len(names)} Python file{'s' if len(names) != 1 else ''}; label: {truth}"
    )
    vc = st.columns(len(MODELS))
    for col, model in zip(vc, MODELS, strict=True):
        called, right_call = verdict(row, model)
        col.markdown(
            f'<div class="verdict">{LABELS[model]}: {word(called)} {"✓" if right_call else "✗"}<br>'
            f"<small>margin {row[f'{model}_margin']:.3g}, cut-off {meta['cutoff'][model]:.3g}</small></div>",
            unsafe_allow_html=True,
        )

    heats = {}
    for model in SHOWN:
        heat, absent = codefeatures.line_heat(phi[model][i], occ["occ"], len(names))
        heats[model] = (heat, absent)
    weight = [sum(abs(v) for m in SHOWN for v in heats[m][0][fi].values()) for fi in range(len(names))]
    order = sorted(range(len(names)), key=lambda fi: -weight[fi])
    if not names:
        st.info("No Python file in this package.")
        st.stop()
    fc, wc = st.columns([3, 1])
    fi = fc.selectbox("File (most attributed first)", order, format_func=lambda k: names[k])
    whole = wc.toggle("Whole file", value=False)
    text = files.get(names[fi], "")
    n_lines = len(text.splitlines())
    both = {
        ln: abs(heats["fptm"][0][fi].get(ln, 0)) + abs(heats["xgboost"][0][fi].get(ln, 0))
        for ln in range(1, n_lines + 1)
    }
    hot = max(both, key=both.get) if both else 1
    start = max(1, hot - WINDOW // 3)
    lines = range(1, min(n_lines, 3000) + 1) if whole else range(start, start + WINDOW)

    cols = st.columns(2)
    for col, model in zip(cols, SHOWN, strict=True):
        heat, absent = heats[model]
        top = max((abs(v) for f in heat for v in f.values()), default=0.0)
        col.markdown(f"**{LABELS[model]}**: {word(verdict(row, model)[0])}", unsafe_allow_html=True)
        col.markdown(code_html(text, heat[fi], top, lines), unsafe_allow_html=True)
        present = sorted(((feats[int(c)], float(phi[model][i][int(c)]), v) for c, v in occ["occ"].items()),
                         key=lambda t: -abs(t[1]))[:8]  # fmt: skip
        table = pd.DataFrame(
            [
                dict(
                    feature=f,
                    attribution=round(a, 3),
                    where=", ".join(sorted({f"{names[fl].rsplit('/', 1)[-1]}:{ln}" for fl, ln in v})[:3])
                    + (" …" if len(v) > 3 else ""),
                )
                for f, a, v in present
            ]  # fmt: skip
        )
        col.dataframe(table, hide_index=True, width="stretch")
        col.caption(f"Features absent from this package contribute {absent:+.3g} in total.")
    st.caption(
        "Red lines push towards malicious, blue towards benign; the stronger the colour, the larger the push "
        "(scaled per model). A line's value is the sum of the attributions of the features on it: each feature's "
        "attribution (FPTM exact Shapley values in closed form; XGBoost interventional TreeSHAP; 100 training "
        "packages as background) is split equally over its occurrences, so every use of a feature counts the same."
    )
