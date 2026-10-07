"""End-to-end: real checkout, retrieval, LangGraph loop and gate; scripted LLM."""
import sys

from app.core import llm
from app.core.config import get_settings
from app.demo import HALLUCINATED_EDIT, scripted_llm
from app.pipeline import solve_issue


def _solve(repo, fn):
    with llm.use_llm(fn):
        return solve_issue("SIP future value is wildly overstated", "annual rate applied monthly",
                           "demo/wealth-lib", repo_path=repo)


def test_demo_converges_after_one_reflection(wealth_repo, monkeypatch):
    monkeypatch.setattr(get_settings(), "gate_test_command", f'"{sys.executable}" -m pytest -q -p no:cacheprovider')
    r = _solve(wealth_repo, scripted_llm)
    assert r.passed and r.gate_passed
    assert r.reflections == 1
    assert [h["gate_passed"] for h in r.history] == [False, True]
    assert r.patch.startswith("--- a/wealth/sip.py")
    assert "-    r = annual_rate_pct / 100\n+    r = monthly_rate(annual_rate_pct)" in r.patch
    assert "search block not found" in r.history[0]["gate_summary"]
    assert r.retrieved_files[0] == "wealth/sip.py"
    assert r.llm_calls == 5  # hyde, planner, engineer, reflector, reviewer (no review of the broken patch)
    assert r.gate["tests_passed"] is True
    assert r.budget_exhausted is False and r.cost_usd == 0.0


def test_never_converging_run_returns_best_attempt_unverified(wealth_repo):
    def fn(role, schema, system, user):
        if role == "reflector":
            return schema(edits=[HALLUCINATED_EDIT], changes_made="nope", new_confidence=0.9)
        return scripted_llm(role, schema, system, user)

    r = _solve(wealth_repo, fn)
    assert not r.passed and not r.gate_passed
    assert r.reflections == get_settings().max_reflections
    assert len(r.history) == get_settings().max_reflections + 1


def test_checkout_is_left_clean(wealth_repo):
    from tests.conftest import git
    _solve(wealth_repo, scripted_llm)
    assert git(wealth_repo, "status", "--porcelain").strip() == ""


def test_cli_demo_exit_code(capsys):
    from app.cli import main
    assert main(["demo"]) == 0
    assert "VERIFIED" in capsys.readouterr().out
