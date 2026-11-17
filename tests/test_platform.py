"""Platform pieces that run without data or services: the state store, months, and the LLM draft check."""

from __future__ import annotations

import pytest

from tmcyberops.platform import config, llm, state


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TMCO_PLATFORM_DIR", str(tmp_path))
    return tmp_path


def test_state_roundtrip(home):
    rid = state.init()
    assert state.init() == rid  # the replay id survives a restart
    state.upsert("months", [{"month": "2013-01", "ts": state.now()}])
    state.upsert("months", [{"month": "2013-01", "ts": state.now()}])  # replaced, not duplicated
    assert state.done_months() == {"2013-01"}
    state.reset()
    assert state.replay_id() is None
    assert state.init() != rid


def test_months_and_previous():
    cfg = {"replay": {"first_month": "2013-11", "last_month": "2014-02"}}
    assert config.months(cfg) == ["2013-11", "2013-12", "2014-01", "2014-02"]
    assert config.previous_month("2014-01") == "2013-12"


def test_llm_draft_check():
    facts = "2014-03, XGBoost (frozen, v0): F1 66.0 at the held-out cut-off; distance 0.169, floor 0.113."
    assert llm.check("XGBoost F1 66.0 in 2014-03; distance 0.169 above 0.113.", facts, 4) is None
    assert "numbers not in the facts" in llm.check("F1 rose to 71.2.", facts, 4)
    assert "lines" in llm.check("a\nb\nc\nd\ne", facts, 4)
    assert llm.check("   ", facts, 4) == "empty"
    # the training year is named in the instructions, so a draft may use it
    assert llm.check("Frozen since 2012, XGBoost fell to 66.0.", llm.given("2014-03", facts), 4) is None
    assert "80" in llm.check("All stayed below 80.", llm.given("2014-03", facts), 4)


def test_llm_retries_a_rejected_draft_once(monkeypatch):
    facts = "XGBoost frozen: F1 66.0 in 2014-03."
    asked = []

    def fake(spec, system, prompt):
        asked.append(prompt)
        return "XGBoost was in the 80s." if len(asked) == 1 else "XGBoost fell to 66.0."

    monkeypatch.setattr(llm, "complete", fake)
    assert llm.draft({}, "2014-03", facts, 4) == {"text": "XGBoost fell to 66.0.", "rejected": None}
    assert len(asked) == 2 and "numbers not in the facts: 80" in asked[1]
    monkeypatch.setattr(llm, "complete", lambda spec, system, prompt: "Around 80.")
    assert "80" in llm.draft({}, "2014-03", facts, 4)["rejected"]  # the reason's numbers are not facts


def test_llm_needs_a_key(monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    cfg = {"llm": {"summarise_with": ["m"], "models": {"m": {"litellm": "x/y", "key_env": "SOME_TEST_KEY"}}}}
    assert llm.available(cfg) == {}
    monkeypatch.setenv("SOME_TEST_KEY", "k")
    assert list(llm.available(cfg)) == ["m"]
