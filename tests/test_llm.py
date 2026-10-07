from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.core import llm


class Out(BaseModel):
    x: int


class _Model:
    def __init__(self, fail, in_tok=100, out_tok=20):
        self.fail, self.in_tok, self.out_tok = fail, in_tok, out_tok

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, messages):
        if self.fail:
            raise TimeoutError("provider down")
        raw = MagicMock(usage_metadata={"input_tokens": self.in_tok, "output_tokens": self.out_tok})
        return {"raw": raw, "parsed": Out(x=7), "parsing_error": None}


@pytest.fixture()
def chain(monkeypatch):
    def install(models: dict):
        monkeypatch.setattr(llm, "chain_specs", lambda cheap=False: list(models))
        monkeypatch.setattr(llm, "build_chat_model", lambda spec, temperature=0.1: models[spec])
    return install


def test_split_spec_keeps_slashes_in_model_id():
    assert llm.split_spec("groq/openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert llm.split_spec("google/gemini-3.6-flash") == ("google", "gemini-3.6-flash")
    assert llm.split_spec("gpt-4o") == ("openai", "gpt-4o")


def test_falls_back_to_next_provider_and_counts_tokens(chain):
    chain({"groq/openai/gpt-oss-120b": _Model(True), "openai/gpt-4o": _Model(False, 1000, 500)})
    with llm.track_usage() as u:
        assert llm.call("planner", "s", "u", schema=Out).x == 7
    assert u.by_provider == {"openai": 1}
    assert len(u.failures) == 1 and "groq" in u.failures[0]
    assert (u.prompt_tokens, u.completion_tokens) == (1000, 500)
    assert u.cost_usd == pytest.approx((1000 * 2.5 + 500 * 10) / 1e6)


def test_free_tier_models_cost_zero(chain):
    chain({"groq/openai/gpt-oss-120b": _Model(False)})
    with llm.track_usage() as u:
        llm.call("planner", "s", "u", schema=Out)
    assert u.total_tokens == 120 and u.cost_usd == 0.0


def test_all_providers_failing_raises(chain):
    chain({"groq/x": _Model(True), "google/y": _Model(True)})
    with pytest.raises(llm.LLMUnavailable, match="All LLM providers failed"):
        llm.call("planner", "s", "u", schema=Out)


def test_unparseable_output_moves_down_the_chain(chain):
    bad = _Model(False)
    bad.invoke = lambda m: {"raw": None, "parsed": None, "parsing_error": "bad json"}
    chain({"groq/x": bad, "google/y": _Model(False)})
    assert llm.call("planner", "s", "u", schema=Out).x == 7


def test_providers_without_keys_are_skipped():
    assert llm.chain_specs() == []  # tests run with no keys
    with pytest.raises(llm.LLMUnavailable, match="GROQ_API_KEY"):
        llm.call("planner", "s", "u", schema=Out)
    assert llm.available() is False


def test_override_receives_role_and_schema():
    seen = []
    with llm.use_llm(lambda role, schema, s, u: seen.append((role, schema)) or Out(x=1)):
        llm.call("reviewer", "s", "u", schema=Out)
        assert llm.available()
    assert seen == [("reviewer", Out)]
