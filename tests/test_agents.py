"""Agent nodes with a scripted LLM (no network)."""
from app.core import llm
from app.agents.graph import finalize, should_reflect
from app.agents.planner import run_planner
from app.agents.reflector import run_reflector
from app.agents.reviewer import run_reviewer
from app.agents.state import initial_state
from app.core.config import get_settings
from app.demo import CORRECT_EDIT, CORRECT_PATCH, HALLUCINATED_EDIT, HALLUCINATED_PATCH


def state(**kw):
    return initial_state(issue_title="SIP overstated", issue_body="uses annual rate monthly",
                         repo_name="demo/wealth-lib", **kw)


def replies(**by_role):
    def fn(role, schema, system, user):
        val = by_role[role]
        return val(schema) if callable(val) else val
    return fn


def test_planner_builds_plan_items():
    fn = replies(planner=lambda S: S(reasoning="r", subtasks=[
        {"description": "a", "file_hint": "x.py"}, {"description": "b"}]))
    with llm.use_llm(fn):
        out = run_planner(state())
    assert [p["description"] for p in out["plan"]] == ["a", "b"]
    assert out["plan"][1]["file_hint"] == "unknown"
    assert all(len(p["id"]) == 8 for p in out["plan"])
    assert out["status"] == "engineering"


def test_planner_caps_at_four_subtasks():
    fn = replies(planner=lambda S: S(reasoning="r", subtasks=[{"description": str(i)} for i in range(9)]))
    with llm.use_llm(fn):
        assert len(run_planner(state())["plan"]) == 4


def test_gate_failure_overrides_a_confident_llm(wealth_repo):
    """Even a 0.99 review cannot pass a patch that doesn't apply — and the LLM isn't even asked."""
    called = []
    fn = replies(reviewer=lambda S: called.append(1) or S(score=0.99, feedback="great"))
    with llm.use_llm(fn):
        out = run_reviewer(state(repo_path=wealth_repo, patch=HALLUCINATED_PATCH))
    assert out["review_passed"] is False and out["review_score"] == 0.0
    assert called == []
    assert any("patch does not apply" in i for i in out["review_issues"])
    assert out["history"][0]["gate_passed"] is False


def test_pass_decision_is_computed_not_trusted(wealth_repo):
    fn = replies(reviewer=lambda S: S(score=0.5, feedback="misses edge case"))
    with llm.use_llm(fn):
        out = run_reviewer(state(repo_path=wealth_repo, patch=CORRECT_PATCH))
    assert out["gate"]["passed"] is True
    assert out["review_passed"] is False  # 0.5 < threshold
    assert out["best_patch"] == CORRECT_PATCH and out["best_gate_passed"] is True


def test_worse_reflection_does_not_replace_best(wealth_repo):
    fn = replies(reviewer=lambda S: S(score=0.6, feedback="ok-ish"))
    with llm.use_llm(fn):
        first = run_reviewer(state(repo_path=wealth_repo, patch=CORRECT_PATCH))
        second = run_reviewer({**first, "patch": HALLUCINATED_PATCH, "reflection_count": 1})
    assert second["best_patch"] == CORRECT_PATCH
    final = finalize(second)
    assert final["patch"] == CORRECT_PATCH and final["review_score"] == 0.6


def test_reflector_gets_gate_errors_and_builds_a_real_diff(wealth_repo):
    seen = {}

    def fn(role, schema, system, user):
        seen["user"] = user
        return schema(edits=[CORRECT_EDIT], changes_made="fixed", new_confidence=0.8)

    with llm.use_llm(fn):
        out = run_reflector(state(repo_path=wealth_repo, edits=[HALLUCINATED_EDIT],
                                  review_issues=["wealth/sip.py: search block not found"]))
    assert "search block not found" in seen["user"] and "rate = annual_rate_pct" in seen["user"]
    assert out["patch"].startswith("--- a/wealth/sip.py")
    assert "+    r = monthly_rate(annual_rate_pct)" in out["patch"]
    assert out["edit_errors"] == [] and out["files_modified"] == ["wealth/sip.py"]
    assert out["reflection_count"] == 1 and out["status"] == "reviewing"


def test_edit_errors_fail_the_gate_without_calling_the_llm():
    called = []
    fn = replies(reviewer=lambda S: called.append(1) or S(score=0.99, feedback="great"))
    with llm.use_llm(fn):
        out = run_reviewer(state(edit_errors=["wealth/sip.py: search block not found"]))
    assert out["review_passed"] is False and called == []
    assert out["review_issues"] == ["wealth/sip.py: search block not found"]


def test_reflector_stops_after_max_rounds():
    n = get_settings().max_reflections
    out = run_reflector(state(reflection_count=n))
    assert out["status"] == "done" and out["reflection_count"] == n + 1


def test_should_reflect_routing():
    n = get_settings().max_reflections
    assert should_reflect(state(review_passed=False, reflection_count=0)) == "reflect"
    assert should_reflect(state(review_passed=True)) == "done"
    assert should_reflect(state(review_passed=False, reflection_count=n)) == "done"


def test_budget_kill_switch_stops_loop(monkeypatch):
    """Token budget exhaustion ends the loop even with reflection rounds left."""
    monkeypatch.setattr(get_settings(), "max_tokens_per_task", 1000)
    with llm.track_usage() as u:
        assert should_reflect(state(review_passed=False, reflection_count=0)) == "reflect"
        u.prompt_tokens = 1000
        assert should_reflect(state(review_passed=False, reflection_count=0)) == "done"
