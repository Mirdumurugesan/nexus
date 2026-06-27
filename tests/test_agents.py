"""
Unit tests for agent logic (mocked LLM calls).
"""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.state import NexusState


def make_state(**overrides) -> NexusState:
    base: NexusState = {
        "task_id": "test-123",
        "issue_title": "Fix proxy handling bug",
        "issue_body": "When HTTPS_PROXY is set, HTTP requests also use it incorrectly.",
        "repo_name": "psf/requests",
        "repo_url": "https://github.com/psf/requests.git",
        "plan": [],
        "plan_reasoning": "",
        "retrieved_context": "def send(self, request, **kwargs):\n    pass",
        "patch": "",
        "patch_explanation": "",
        "files_modified": [],
        "confidence": 0.0,
        "root_cause": "",
        "review_score": 0.0,
        "review_feedback": "",
        "review_passed": False,
        "reflection_count": 0,
        "error": "",
        "status": "planning",
    }
    base.update(overrides)
    return base


class TestPlannerAgent:
    def test_planner_adds_plan_to_state(self):
        from app.agents.planner import run_planner

        mock_output = MagicMock()
        mock_output.reasoning = "The proxy merge logic incorrectly applies HTTPS_PROXY to HTTP."
        mock_output.subtasks = [
            {"description": "Fix merge_environment_settings", "file_hint": "requests/adapters.py"},
            {"description": "Add test for HTTP-only proxy", "file_hint": "tests/test_proxies.py"},
        ]

        with patch("app.agents.planner.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output
            state = make_state()
            result = run_planner(state)

        assert len(result["plan"]) == 2
        assert result["plan_reasoning"] != ""
        assert result["status"] == "engineering"

    def test_planner_plan_has_required_fields(self):
        from app.agents.planner import run_planner

        mock_output = MagicMock()
        mock_output.reasoning = "Root cause."
        mock_output.subtasks = [
            {"description": "Fix the bug", "file_hint": "main.py"},
        ]

        with patch("app.agents.planner.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output
            result = run_planner(make_state())

        plan_item = result["plan"][0]
        assert "id" in plan_item
        assert "description" in plan_item
        assert "file_hint" in plan_item
        assert "status" in plan_item


class TestReviewerAgent:
    def test_reviewer_passes_good_patch(self):
        from app.agents.reviewer import run_reviewer

        mock_output = MagicMock()
        mock_output.score = 0.85
        mock_output.passed = True
        mock_output.feedback = ""
        mock_output.issues_found = []

        with patch("app.agents.reviewer.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output
            state = make_state(patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new")
            result = run_reviewer(state)

        assert result["review_score"] == 0.85
        assert result["review_passed"] is True
        assert result["status"] == "done"

    def test_reviewer_fails_bad_patch(self):
        from app.agents.reviewer import run_reviewer

        mock_output = MagicMock()
        mock_output.score = 0.4
        mock_output.passed = False
        mock_output.feedback = "Patch is incomplete."
        mock_output.issues_found = ["Missing edge case"]

        with patch("app.agents.reviewer.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output
            result = run_reviewer(make_state(patch="minimal patch"))

        assert result["review_passed"] is False
        assert result["status"] == "reflecting"


class TestReflectorAgent:
    def test_reflector_improves_patch(self):
        from app.agents.reflector import run_reflector

        mock_output = MagicMock()
        mock_output.improved_patch = "--- a/fix.py\n+++ b/fix.py\n@@ -1 +1 @@\n+improved"
        mock_output.changes_made = "Added missing edge case handling."
        mock_output.new_confidence = 0.82

        with patch("app.agents.reflector.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output
            state = make_state(
                patch="old patch",
                review_score=0.4,
                review_feedback="Incomplete",
                reflection_count=0,
            )
            result = run_reflector(state)

        assert result["patch"] == mock_output.improved_patch
        assert result["confidence"] == 0.82
        assert result["reflection_count"] == 1
        assert result["status"] == "reviewing"

    def test_reflector_stops_at_max_reflections(self):
        from app.agents.reflector import run_reflector, MAX_REFLECTIONS

        state = make_state(reflection_count=MAX_REFLECTIONS)
        result = run_reflector(state)

        assert result["status"] == "done"
        assert result["reflection_count"] == MAX_REFLECTIONS + 1


class TestGraphConditionalEdge:
    def test_should_reflect_when_failed(self):
        from app.agents.graph import should_reflect
        state = make_state(review_passed=False, reflection_count=0)
        assert should_reflect(state) == "reflect"

    def test_should_be_done_when_passed(self):
        from app.agents.graph import should_reflect
        state = make_state(review_passed=True, reflection_count=0)
        assert should_reflect(state) == "done"

    def test_should_be_done_after_max_reflections(self):
        from app.agents.graph import should_reflect
        from app.agents.reflector import MAX_REFLECTIONS
        state = make_state(review_passed=False, reflection_count=MAX_REFLECTIONS)
        assert should_reflect(state) == "done"
