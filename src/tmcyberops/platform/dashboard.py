"""Streamlit dashboard of the replay.

Overview: status, a decision if one is waiting, the whole replay as a heatmap, detection, learning
gain and drift per model (one line per learning arm), and the events on a timeline. Month report: the
month in brief (by language models when configured) and what the model looks at. Started by
`tmco-up`; refreshes itself.

Playback, for recordings: http://localhost:8501/?playback=2 shows the finished months again from the
first one, one more every 2 seconds. Nothing is recomputed; the stored results are revealed in order.
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tmcyberops.datasets.drebin_labels import label as feature_label
from tmcyberops.platform import config, state
from tmcyberops.platform.config import MODEL_LABELS

CFG = config.load()
MONTHS = config.months(CFG)
MODELS = CFG["models"]
ARMS = CFG["arms"]
DATASETS = {
    "apigraph": "APIGraph · Android apps",
    "bodmas": "BODMAS · Windows programs",
    "anoshift": "AnoShift · network traffic",
    "phreshphish": "PhreshPhish · web pages",
    "beth": "BETH · host events",
}
ARM_TEXT = {
    "frozen": "Frozen (2012)",
    "retrain": "Retrained on approval",
    "incremental": "Learns every month",
}
ARM_SHORT = {"frozen": "frozen", "retrain": "retrained", "incremental": "learning every month"}
ARM_BUTTON = {"frozen": "Frozen", "retrain": "Retrained", "incremental": "Learning"}
# Categorical slots 1-3 of the validated reference palette; an arm keeps its colour everywhere.
ARM_COLORS = {
    "light": {"frozen": "#2a78d6", "retrain": "#eb6834", "incremental": "#1baf7a"},
    "dark": {"frozen": "#3987e5", "retrain": "#d95926", "incremental": "#199e70"},
}
# Sequential blue ramp of the same palette; on the dark surface low values recede into the background.
RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec",
        "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]  # fmt: skip
STATUS = {"warning": "#fab219", "serious": "#ec835a", "good": "#0ca30c"}
MUTED = {"light": "#52514e", "dark": "#9aa7bd"}
GRID = {"light": "rgba(11,11,11,0.07)", "dark": "rgba(154,167,189,0.13)"}
CSS = """<style>
.legend {display:flex; gap:1.6rem; flex-wrap:wrap; margin:0.2rem 0 0.4rem 0; font-size:0.92rem}
.legend span.sw {display:inline-block; width:22px; height:0; border-top:3px solid; border-radius:2px;
                 vertical-align:middle; margin-right:0.45rem}
.feed p {margin:0.15rem 0}
.feed h6 {margin:0.9rem 0 0.2rem 0; font-size:0.95rem}
.muted {opacity:0.7}
.tile {padding:0.7rem 0.9rem; border-radius:10px; background:rgba(127,156,199,0.10)}
.tile .t {font-size:0.82rem; opacity:0.75}
.tile .v {font-size:1.8rem; font-weight:600; line-height:1.25}
.tile .s {font-size:0.82rem; opacity:0.75}
</style>"""

st.set_page_config(page_title="Replay monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
try:
    PLAY = max(0.5, float(st.query_params.get("playback", "0"))) if st.query_params.get("playback") else 0.0
except ValueError:
    PLAY = 0.0


# ---------------------------------------------------------------- small helpers


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def month_name(m: str, short: bool = False) -> str:
    return datetime.strptime(m, "%Y-%m").strftime("%b %Y" if short else "%B %Y")


def ts(m) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(m + "-01") if isinstance(m, pd.Series) else pd.Timestamp(m + "-01")


def name(model: str) -> str:
    """Model name inside a sentence."""
    return "logistic regression" if model == "logreg" else MODEL_LABELS[model]


def and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def period_text(trained_on: str) -> str:
    """'2012 pool + 2013-01..2013-02' -> '2012 + Jan 2013 to Feb 2013'."""
    out = trained_on.replace("2012 pool", "2012")
    for a, b in re.findall(r"(\d{4}-\d{2})\.\.(\d{4}-\d{2})", out):
        out = out.replace(f"{a}..{b}", f"{month_name(a, True)} to {month_name(b, True)}")
    return out


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def tile(col, title: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="tile"><div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def legend(pal: dict) -> None:
    items = "".join(
        f'<span><span class="sw" style="border-color:{pal[a]}"></span>{ARM_TEXT[a]}</span>' for a in ARMS
    )
    st.markdown(f'<div class="legend">{items}</div>', unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", theme="streamlit")


def base_layout(fig: go.Figure, height: int, t: str) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=28, b=4),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor=GRID[t], zeroline=False)
    return fig


def x_axis(fig: go.Figure, last: str) -> None:
    """The x axis grows with the replay: at least a year, three months beyond the last finished one."""
    end = min(len(MONTHS) - 1, max(MONTHS.index(last) + 3, 11))
    short = end < 24
    fig.update_xaxes(
        range=[ts(MONTHS[0]), ts(MONTHS[end])],
        dtick="M3" if short else "M12",
        tickformat="%b %Y" if short else "%Y",
        showgrid=False,
    )


# ---------------------------------------------------------------- charts


def arm_rows(df: pd.DataFrame, model: str, arm: str) -> pd.DataFrame:
    """The rows of one line. The retrained arm equals the frozen one until a retrained model is in use,
    so its line starts there, joined to the frozen line in the month before."""
    g = df[(df["model"] == model) & (df["arm"] == arm)].sort_values("month")
    if arm != "retrain":
        return g
    newer = g[g["version"] > 0]
    if not len(newer):
        return g.iloc[0:0]
    first = newer["month"].iloc[0]
    i = MONTHS.index(first)
    before = (
        df[(df["model"] == model) & (df["arm"] == "frozen") & (df["month"] == MONTHS[i - 1])] if i else None
    )
    return pd.concat([before, g[g["month"] >= first]]) if before is not None else g[g["month"] >= first]


def model_panels() -> go.Figure:
    return make_subplots(
        rows=1,
        cols=len(MODELS),
        shared_yaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=[MODEL_LABELS[m] for m in MODELS],
    )


def lines_chart(
    df, col, pal, t, y_title, hover, sel, last, y_range=None, line_at=None, line_text="", marks=None
):
    """One panel per model, one line per arm, shared y axis."""
    fig = model_panels()
    for c, model in enumerate(MODELS, 1):
        for arm in sorted(ARMS, key=lambda a: a == "frozen"):  # frozen drawn last, on top
            g = arm_rows(df, model, arm)
            fig.add_trace(
                go.Scatter(
                    x=ts(g["month"]),
                    y=g[col],
                    mode="lines",
                    line=dict(color=pal[arm], width=2.2),
                    hovertemplate=f"{ARM_TEXT[arm]}: %{{y:{hover}}}<extra></extra>",
                ),
                row=1,
                col=c,
            )
        if marks is not None and len(marks[marks["model"] == model]):
            g = marks[marks["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=ts(g["month"]),
                    y=g[col],
                    mode="markers",
                    marker=dict(
                        symbol="diamond", size=12, color=pal["retrain"], line=dict(width=2, color="white")
                    ),
                    hovertemplate="retrained model in use<extra></extra>",
                ),
                row=1,
                col=c,
            )
        if line_at is not None:
            fig.add_hline(y=line_at, line=dict(color=MUTED[t], width=1), row=1, col=c)
        fig.add_vline(x=ts(sel), line=dict(color=MUTED[t], width=1, dash="dot"), row=1, col=c)
    if line_at is not None and line_text:
        n = len(MODELS)
        end = min(len(MONTHS) - 1, max(MONTHS.index(last) + 3, 11))
        fig.add_annotation(
            x=ts(MONTHS[end]),
            y=line_at,
            xref=f"x{n}",
            yref=f"y{n}",
            text=line_text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color=MUTED[t]),
        )
    base_layout(fig, 270, t)
    x_axis(fig, last)
    fig.update_yaxes(range=y_range)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    return fig


def gain_chart(metrics: pd.DataFrame, pal: dict, t: str, sel: str, last: str) -> go.Figure:
    """F1 of each learning arm minus F1 of the frozen model, month by month, as shaded areas."""
    fig = model_panels()
    for c, model in enumerate(MODELS, 1):
        frozen = metrics[(metrics["model"] == model) & (metrics["arm"] == "frozen")].set_index("month")["f1"]
        for arm in [a for a in ARMS if a != "frozen"]:
            g = arm_rows(metrics, model, arm).set_index("month")
            if not len(g):
                continue
            gain = (g["f1"] - frozen.reindex(g.index)).fillna(0.0)
            fig.add_trace(
                go.Scatter(
                    x=ts(pd.Series(gain.index)),
                    y=gain.values,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=pal[arm], width=1.8),
                    fillcolor=rgba(pal[arm], 0.22),
                    hovertemplate=f"{ARM_TEXT[arm]}: %{{y:+.1f}} F1<extra></extra>",
                ),
                row=1,
                col=c,
            )
        fig.add_hline(y=0, line=dict(color=MUTED[t], width=1), row=1, col=c)
        fig.add_vline(x=ts(sel), line=dict(color=MUTED[t], width=1, dash="dot"), row=1, col=c)
    base_layout(fig, 230, t)
    x_axis(fig, last)
    fig.update_yaxes(title_text="F1 vs frozen", row=1, col=1)
    return fig


def heatmap(metrics: pd.DataFrame, t: str) -> go.Figure:
    """Every version x every month of the replay; empty cells are months still to come."""
    rows = [(m, a) for m in MODELS for a in ARMS]
    labels = [f"{MODEL_LABELS[m]} · {ARM_BUTTON[a].lower()}" for m, a in rows]
    z = np.full((len(rows), len(MONTHS)), np.nan)
    piv = metrics.pivot_table(index=["model", "arm"], columns="month", values="f1")
    for i, key in enumerate(rows):
        if key in piv.index:
            for m, v in piv.loc[key].items():
                z[i, MONTHS.index(m)] = v
    ramp = RAMP if t == "dark" else RAMP[::-1]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[ts(m) for m in MONTHS],
            y=labels,
            colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
            zmin=50,
            zmax=100,
            xgap=1,
            ygap=3,
            hovertemplate="%{y}<br>%{x|%b %Y}: F1 %{z:.1f}<extra></extra>",
            colorbar=dict(title=dict(text="F1", side="top"), thickness=10, len=0.9, outlinewidth=0),
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(dtick="M12", tickformat="%Y", showgrid=False)
    return fig


EVENT_STYLE = {
    "drift": ("triangle-up", "warning", "drift alert"),
    "f1": ("triangle-down", "serious", "detection alert"),
    "retrain proposed": ("diamond-open", "retrain", "retrain proposed"),
    "retrain promoted": ("diamond", "retrain", "retrained model in use"),
    "retrain rejected": ("x-thin-open", "muted", "rejected"),
    "drift cleared": ("circle-open", "muted", "alert cleared"),
    "f1 cleared": ("circle-open", "muted", "alert cleared"),
}


def timeline(events: pd.DataFrame, pal: dict, t: str, last: str) -> go.Figure:
    fig = go.Figure()
    color = {
        "warning": STATUS["warning"],
        "serious": STATUS["serious"],
        "retrain": pal["retrain"],
        "muted": MUTED[t],
    }
    shown = set()
    for kind, (symbol, col, text) in EVENT_STYLE.items():
        e = events[events["kind"] == kind]
        if not len(e):
            continue
        fig.add_trace(
            go.Scatter(
                x=ts(e["month"]),
                y=[MODELS.index(m) for m in e["model"]],
                mode="markers",
                name=text,
                showlegend=text not in shown,
                legendgroup=text,
                marker=dict(symbol=symbol, size=13, color=color[col], line=dict(width=1.5, color=color[col])),
                customdata=np.stack([e["arm"].map(ARM_SHORT), e["model"].map(MODEL_LABELS)], axis=-1),
                hovertemplate=f"{text} · %{{customdata[1]}}, %{{customdata[0]}}<extra></extra>",
            )
        )
        shown.add(text)
    fig.update_layout(
        height=210,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
        hovermode="closest",
    )
    rows = len(MODELS)
    fig.update_yaxes(
        tickvals=list(range(rows)),
        ticktext=[MODEL_LABELS[m] for m in MODELS],
        range=[rows - 0.5, -0.5],
        gridcolor=GRID[t],
        zeroline=False,
    )
    x_axis(fig, last)
    return fig


def slope_chart(f: pd.DataFrame, pal: dict, t: str, sel: str, arm: str) -> go.Figure:
    """Rank in the training data against rank this month, for the features that moved most."""
    fig = go.Figure()
    up = f[f["kind"] == "entered"].head(4)
    down = f[f["kind"] == "left"].head(4)
    x = ["training data", month_name(sel, True)]
    for rows, colr, width in ((down, MUTED[t], 1.5), (up, pal[arm], 2.5)):
        for _, r in rows.iterrows():
            _, text = feature_label(r["feature"])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[r["reference_rank"], r["rank"]],
                    mode="lines+markers+text",
                    line=dict(color=colr, width=width),
                    marker=dict(size=8, color=colr),
                    text=["", text],
                    textposition="middle right",
                    textfont=dict(size=11),
                    hovertemplate=f"{html.escape(text)}: rank %{{y}}<extra></extra>",
                )
            )
    fig.update_layout(
        height=300,
        margin=dict(l=4, r=170, t=8, b=4),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(type="log", autorange="reversed", title_text="rank (1 = most weight)", gridcolor=GRID[t])
    fig.update_xaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------- text


def brief(month: str, metrics: pd.DataFrame, drift: pd.DataFrame) -> list[str]:
    """The month in a few sentences, from the numbers (used when no language model is configured)."""
    m = metrics[metrics["month"] == month]
    if not len(m):
        return []
    best = m.loc[m["f1"].idxmax()]
    out = [f"**Best detection:** {name(best['model'])}, {ARM_SHORT[best['arm']]} (F1 {best['f1']:.1f})."]
    gains = []
    for model in MODELS:
        g = m[m["model"] == model].set_index("arm")["f1"]
        if "frozen" in g and len(g) > 1:
            other = g.drop("frozen")
            gains.append((other.max() - g["frozen"], model, other.idxmax(), other.max(), g["frozen"]))
    if gains:
        d, model, arm, val, base = max(gains)
        if d >= 1:
            out.append(
                f"**Learning helps most for {name(model)}:** {ARM_SHORT[arm]} {val:.1f}, frozen {base:.1f}."
            )
        else:
            out.append(
                "**Learning makes little difference so far:** every arm is within 1 F1 point of frozen."
            )
    d = drift[(drift["month"] == month) & (drift["arm"] == "frozen")]
    if len(d):
        above = [name(x) for x in d[d["above"]]["model"]]
        below = [name(x) for x in d[~d["above"]]["model"]]
        if above:
            verb = "is" if len(above) == 1 else "are"
            tail = f"; {and_list(below)} {'is' if len(below) == 1 else 'are'} not" if below else ""
            out.append(f"**Drift:** the frozen {and_list(above)} {verb} above the noise floor{tail}.")
        else:
            out.append("**Drift:** no frozen model is above its noise floor.")
    return out


def event_sentence(r: pd.Series) -> tuple[str, str]:
    """(icon, sentence) for one event."""
    model = MODEL_LABELS.get(r["model"], r["model"])
    arm, msg, kind = ARM_SHORT.get(r["arm"], r["arm"]), r["message"], r["kind"]
    v = re.search(r"\bv(\d+)\b", msg)
    ver = f" v{v.group(1)}" if v else ""
    if kind == "drift":
        return "⚠️", f"{model} ({arm}) drifts: its features moved beyond the noise floor two months running."
    if kind == "drift cleared":
        return "↩️", f"{model} ({arm}) is back within its noise floor."
    if kind == "f1":
        f = re.search(r"F1 (\d+\.\d)", msg)
        return "⚠️", f"{model} ({arm}) detection fell to F1 {f.group(1) if f else '?'}, below its alert line."
    if kind == "f1 cleared":
        return "↩️", f"{model} ({arm}) detection recovered."
    if kind == "retrain proposed":
        t = re.search(r"trained on (.+?) \(", msg)
        period = period_text(t.group(1)) if t else "recent months"
        return "⏳", f"New {model} model{ver}, trained on {period}, waits for approval."
    if kind == "retrain promoted":
        how = "approved in n8n" if "n8n" in msg else "approved automatically after the timeout"
        return "✅", f"The retrained {model}{ver} is now in use ({how})."
    if kind == "retrain rejected":
        return "✖️", f"{model}{ver} was rejected; the current model stays."
    return "•", msg


def feature_block(f: pd.DataFrame) -> str:
    def items(kind: str, n: int, ranks: bool) -> str:
        out = []
        for _, r in f[f["kind"] == kind].head(n).iterrows():
            cat, text = feature_label(r["feature"])
            rk = f" <span class='muted'>· rank {r['reference_rank']} → {r['rank']}</span>" if ranks else ""
            raw = html.escape(r["feature"], quote=True)
            out.append(
                f"<p title='{raw}'><b>{html.escape(text)}</b> <span class='muted'>· {cat}</span>{rk}</p>"
            )
        return "".join(out) or "<p class='muted'>none</p>"

    return (
        "<div class='feed'>"
        f"<h6>Carry most of the malware score now</h6>{items('top', 5, False)}"
        f"<h6>Gained weight since the training data</h6>{items('entered', 4, True)}"
        f"<h6>Lost weight</h6>{items('left', 4, True)}"
        "</div>"
    )


# ---------------------------------------------------------------- page


def decision_cards(models: pd.DataFrame, events: pd.DataFrame) -> None:
    for _, r in models[models["status"] == "pending"].iterrows():
        ev = events[(events["kind"] == "retrain proposed") & (events["model"] == r["model"])]
        link = ev["link"].iloc[0] if len(ev) else None
        with st.container(border=True):
            a, b = st.columns([5, 1], vertical_alignment="center")
            a.markdown(
                f"⏳ **Decision needed.** A new {name(r['model'])} model is ready, trained on "
                f"{period_text(r['trained_on'])} (F1 {r['heldout_f1_cutoff']:.1f} on its held-out apps). It replaces "
                f"the retrained {name(r['model'])} when you approve, or by itself after "
                f"{CFG['retrain']['approval_timeout_minutes']} minutes."
            )
            if isinstance(link, str) and link:
                b.link_button("Approve or reject", link, type="primary")
            else:
                b.caption("n8n not reachable")


def playback(done: list[str]) -> list[str]:
    """The finished months revealed so far: one at the start, one more every PLAY seconds."""
    start = st.session_state.setdefault("play_start", time.time())
    return done[: max(1, min(len(done), 1 + int((time.time() - start) / PLAY)))]


def touched() -> None:
    st.session_state["month_touched"] = True


@st.fragment(run_every=PLAY or 15)
def page() -> None:
    t = theme()
    pal = ARM_COLORS[t]
    head, pick = st.columns([3, 2], vertical_alignment="bottom")
    head.title("Replay monitor")
    dataset = pick.selectbox("Dataset", list(DATASETS), format_func=DATASETS.get, key="dataset")
    if dataset != "apigraph":
        st.info(
            f"{DATASETS[dataset]} is not connected yet: the platform replays APIGraph only. Its data adapter "
            "comes first; the dashboard then shows it with the same charts."
        )
        return
    st.caption(
        "Android apps from 2013 to 2018, one month at a time. Three detectors, each in three versions: "
        "frozen as trained on 2012, retrained when you approve, and one that learns every month."
    )
    if not state.db_path().exists() or not state.models_ready():
        st.info("Waiting for the platform: start it with `uv run tmco-up`; the 2012 models train first.")
        return
    finished = sorted(state.done_months())
    if not finished:
        st.info("The 2012 models are ready; the first month is running.")
        return
    done = playback(finished) if PLAY else finished
    rule = st.session_state.get("rule") or "held-out"
    # only finished months: a month still running (or stopped half-way) has partial rows
    metrics = state.query(
        "select * from metrics where rule = ? and month in (select month from months) order by month", [rule]
    )
    drift = state.query(
        "select * from drift where versus = 'reference' and month in (select month from months) order by month"
    )
    drift["ratio"] = drift["distance"] / drift["floor_p95"]
    models = state.query("select * from models")
    events = state.query("select * from events order by ts desc")
    latest = done[-1]
    if PLAY:
        metrics = metrics[metrics["month"] <= latest]
        drift = drift[drift["month"] <= latest]
        events = events[events["month"] <= latest]
        models = models[(models["status"] != "pending") | (models["proposed_month"] <= latest)]
        running = len(done) < len(finished)
    else:
        last_ts = state.query("select max(ts) as ts from months")["ts"][0]
        running = last_ts is not None and (state.now() - last_ts).total_seconds() < 600

    c1, c2, c3, c4 = st.columns(4)
    tile(
        c1,
        "Replay " + ("finished" if len(done) == len(MONTHS) else "running" if running else "paused"),
        month_name(latest),
        f"month {len(done)} of {len(MONTHS)}",
    )
    m_last = metrics[metrics["month"] == latest]
    if len(m_last):
        b = m_last.loc[m_last["f1"].idxmax()]
        tile(
            c2,
            "Best detection this month",
            f"F1 {b['f1']:.1f}",
            f"{MODEL_LABELS[b['model']]}, {ARM_SHORT[b['arm']]}",
        )
    d_last = drift[drift["month"] == latest]
    tile(c3, "Drifting", f"{int(d_last['above'].sum())} of {len(d_last)}", "versions above their noise floor")
    pending = models[models["status"] == "pending"]
    tile(
        c4,
        "Your decisions",
        f"{len(pending)} waiting" if len(pending) else "none",
        "retrained models to approve",
    )
    decision_cards(models, events)

    st.write("")
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    # the slider follows the newest month until it is moved by hand
    if st.session_state.get("month") not in done or not st.session_state.get("month_touched"):
        st.session_state["month"] = done[-1]
    sel = left.select_slider("Month", options=done, format_func=month_name, key="month", on_change=touched)
    right.segmented_control(
        "Cut-off",
        ["held-out", "default"],
        default="held-out",
        key="rule",
        format_func=lambda r: "chosen on held-out apps" if r == "held-out" else "default (score > 0)",
    )

    overview, report = st.tabs(["Overview", f"Month report · {month_name(sel)}"])
    with overview:
        st.subheader("The whole replay at a glance")
        show(heatmap(metrics, t))
        st.caption(
            "F1 of every version in every month; brighter is better. Empty cells: months still to come."
        )

        legend(pal)
        st.subheader("Detection: how well each version finds malware")
        promoted = models[(models["arm"] == "retrain") & (models["version"] > 0)]
        promoted = promoted[promoted["status"].isin(["active", "retired"])]
        marks = None
        if len(promoted):
            firsts = metrics[metrics["arm"] == "retrain"].merge(
                promoted[["model", "version"]], on=["model", "version"]
            )
            marks = firsts.sort_values("month").groupby(["model", "version"]).head(1)
        show(lines_chart(metrics, "f1", pal, t, "F1 (%)", ".1f", sel, latest, marks=marks))
        st.caption("F1 of the malware class each month. Diamonds: a retrained model goes into use.")

        st.subheader("Learning gain: what updating buys over the frozen model")
        show(gain_chart(metrics, pal, t, sel, latest))
        st.caption("F1 points above (or below) the frozen model of the same detector.")

        st.subheader("Drift: does each version still see the apps as in its training data?")
        show(
            lines_chart(
                drift,
                "ratio",
                pal,
                t,
                "drift ÷ noise floor",
                ".2f",
                sel,
                latest,
                [0, None],
                1.0,
                "noise floor",
            )
        )
        st.caption(
            "Above 1: the features that carry the model's score changed more than sampling noise explains "
            "(top 100 features of 500 apps, against the same number of apps from the training data)."
        )

        st.subheader("Events")
        if len(events):
            show(timeline(events, pal, t, latest))
            for _, r in events.head(6).iterrows():
                icon, text = event_sentence(r)
                st.markdown(f"{icon} **{month_name(r['month'], True)}** · {text}")
        else:
            st.caption("Nothing yet: alerts, proposals and approvals appear here.")

    with report:
        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("In brief")
            llm = state.query(
                "select author, text from summaries where month = ? and model = 'all' order by author", [sel]
            )
            if len(llm):
                for tab, (_, r) in zip(st.tabs(llm["author"].tolist()), llm.iterrows(), strict=True):
                    for line in [x for x in r["text"].splitlines() if x.strip()]:
                        tab.markdown(line)
                    tab.caption(
                        f"Written by {r['author']} from this month's numbers; numbers checked against them."
                    )
            else:
                for line in brief(sel, metrics, drift):
                    st.markdown(f"- {line}")
            for _, r in events[events["month"] == sel].iterrows():
                icon, text = event_sentence(r)
                st.markdown(f"{icon} {text}")
        with b:
            st.subheader("What the model looks at")
            x, y = st.columns(2)
            fm = x.segmented_control(
                "Model", MODELS, default=MODELS[0], key="fmodel", format_func=MODEL_LABELS.get
            )
            fa = y.segmented_control(
                "Version", ARMS, default="frozen", key="farm", format_func=ARM_BUTTON.get
            )
            f = state.query(
                "select kind, position, feature, rank, reference_rank from features "
                "where month = ? and model = ? and arm = ? order by kind, position",
                [sel, fm or MODELS[0], fa or "frozen"],
            )
            if len(f):
                show(slope_chart(f, pal, t, sel, fa or "frozen"))
                st.caption(
                    "Rank of the features that moved most: coloured = gained weight, grey = lost weight."
                )
            st.markdown(feature_block(f), unsafe_allow_html=True)

        with st.expander("Data tables (for checking)"):
            st.dataframe(models, hide_index=True, width="stretch")
            st.dataframe(metrics[metrics["month"] == sel], hide_index=True, width="stretch")
            st.dataframe(events, hide_index=True, width="stretch")


page()
