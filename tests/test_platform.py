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


def test_llm_needs_a_key(monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    cfg = {"llm": {"summarise_with": ["m"], "models": {"m": {"litellm": "x/y", "key_env": "SOME_TEST_KEY"}}}}
    assert llm.available(cfg) == {}
    monkeypatch.setenv("SOME_TEST_KEY", "k")
    assert list(llm.available(cfg)) == ["m"]
