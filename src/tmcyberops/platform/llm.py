"""Month summaries written by language models, provider-agnostic.

A LangGraph graph fans the month's facts out to every configured model. A model is a name in
ops/platform.yaml (`llm.models`) with a transport:

* `<provider>/<model>`: called through LiteLLM, with its key in an environment variable from the local,
  untracked `.env` (any provider LiteLLM supports: Anthropic, Moonshot, DashScope, ...);
* `claude-cli/<model>`: the Claude Code CLI run headless, so a Claude subscription can stand in until
  API keys are set (CLAUDE_CLI in `.env` is the path to the executable).

Without any configured model the platform keeps the template summary. Each draft is checked before it
is stored: at most `max_lines` lines, and every number in it must occur in the facts the model was
given. A draft that fails the check is stored with the reason, not shown as a summary.
"""

from __future__ import annotations

import json
import operator
import os
import re
import subprocess
import tempfile
from typing import Annotated, TypedDict

from tmcyberops.paths import REPO

# LiteLLM's bundled price table; it would otherwise fetch one from GitHub at import
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

SYSTEM = (
    "You summarise one month of an Android malware detection replay for a security analyst. "
    "Write at most {max_lines} short sentences of plain prose, one per line, no headings, no bullet "
    "symbols, no bold. Give the gist, not a list of numbers: at most five numbers in total. "
    "Use only the facts given; copy numbers exactly as written and add no others. Compare the three "
    "versions of each detector (frozen since 2012, retrained on approval, learning every month) and the "
    "detectors (FPTM, XGBoost, logistic regression); say what changed and what matters. When you name "
    "features, say they contributed most to the model's malware score; never call them malware indicators."
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def prompt(month: str, facts: str) -> str:
    return f"Month {month}.\n\n{facts}"


def given(month: str, facts: str) -> str:
    """Everything the model was told: the instructions (which name the training year) and the facts."""
    return SYSTEM + "\n" + prompt(month, facts)


class Summaries(TypedDict):
    month: str
    facts: str
    drafts: Annotated[dict, operator.or_]


def available(cfg: dict) -> dict[str, dict]:
    """The configured models whose key (or CLI path) is set, after reading the untracked .env."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    llm = cfg.get("llm") or {}
    use = llm.get("summarise_with") or []
    models = llm.get("models") or {}
    return {n: models[n] for n in use if n in models and os.environ.get(models[n].get("key_env", ""), "")}


def check(text: str, facts: str, max_lines: int) -> str | None:
    """None if the draft passes, else the reason it fails."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    if len(lines) > max_lines:
        return f"{len(lines)} lines (at most {max_lines})"
    known = set(NUMBER.findall(facts))
    unknown = sorted(set(NUMBER.findall(text)) - known)
    if unknown:
        return "numbers not in the facts: " + ", ".join(unknown[:5])
    return None


def claude_cli(model: str, system: str, prompt: str, timeout: float) -> str:
    """One answer from the Claude Code CLI, headless, with no tools and no project settings."""
    cmd = [
        os.environ["CLAUDE_CLI"], "-p", "--model", model, "--system-prompt", system, "--tools", "",
        "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
        "--disable-slash-commands", "--output-format", "json",
    ]  # fmt: skip
    r = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=tempfile.gettempdir(), timeout=timeout,
    )  # fmt: skip
    out = json.loads(r.stdout or "{}")
    if r.returncode != 0 or out.get("is_error"):
        raise RuntimeError(f"claude CLI failed: {(out.get('result') or r.stderr)[-300:]}")
    return out.get("result", "")


def complete(spec: dict, system: str, prompt: str) -> str:
    timeout = float(spec.get("timeout_s", 60))
    transport, _, model = spec["litellm"].partition("/")
    if transport == "claude-cli":
        return claude_cli(model, system, prompt, timeout)
    import litellm

    resp = litellm.completion(
        model=spec["litellm"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=int(spec.get("max_tokens", 400)),
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


def _node(name: str, spec: dict, max_lines: int):
    def run(s: Summaries) -> dict:
        try:
            text = complete(spec, SYSTEM.format(max_lines=max_lines), prompt(s["month"], s["facts"]))
            text = text.strip()
        except Exception as e:  # a provider error must not fail the month
            return {"drafts": {name: {"text": "", "rejected": f"call failed: {type(e).__name__}: {e}"[:300]}}}
        return {
            "drafts": {
                name: {"text": text, "rejected": check(text, given(s["month"], s["facts"]), max_lines)}
            }
        }

    return run


def summarise(cfg: dict, month: str, facts: str) -> dict[str, dict]:
    """{model name: {"text", "rejected"}} for every available model; {} when none is configured."""
    models = available(cfg)
    if not models:
        return {}
    from langgraph.graph import END, START, StateGraph

    max_lines = int((cfg.get("llm") or {}).get("max_lines", 4))
    g = StateGraph(Summaries)
    for name, spec in models.items():
        g.add_node(name, _node(name, spec, max_lines))
        g.add_edge(START, name)
        g.add_edge(name, END)
    out = g.compile().invoke({"month": month, "facts": facts, "drafts": {}})
    return out["drafts"]
