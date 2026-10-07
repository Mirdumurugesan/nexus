"""
Reviewer Agent — a two-stage quality gate.

Stage 1 (deterministic): the Patch Gate. Does the diff apply to the real
checkout? Does every touched .py file still compile? Do the tests pass (if a
test command is configured)? If not, the score is 0 and the exact error goes
to the Reflector. No LLM call is spent reviewing a patch that cannot apply.

Stage 2 (LLM): only for patches that survive stage 1 — correctness,
completeness, safety, style. The pass/fail decision is computed here from the
score and the gate, never taken from the model's own "passed" opinion.

The reviewer also keeps the best attempt seen so far, so a reflection round
that makes the patch worse can't overwrite a better earlier one.
"""
import logging
from pydantic import BaseModel, Field

from app.core import llm
from app.agents.state import Attempt, NexusState
from app.core.config import get_settings
from app.tools.patch_gate import GateResult, check_patch, files_in_patch

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM = """You are a strict senior code reviewer.
The patch below already applies cleanly and compiles. Judge whether it actually
fixes the issue.

Score 0.0-1.0 on: correctness (fixes the root cause), completeness (edge cases),
safety (no regressions), style (matches conventions). Below 0.7 = needs work.
Give specific, actionable feedback."""


class ReviewOutput(BaseModel):
    score: float = Field(description="Overall quality score 0.0-1.0", ge=0.0, le=1.0)
    feedback: str = Field(default="", description="Specific, actionable feedback")
    issues_found: list[str] = Field(default_factory=list, description="Concrete problems (empty if none)")


def _gate(state: NexusState) -> GateResult:
    if state.get("edit_errors"):
        return GateResult(passed=False, errors=list(state["edit_errors"]))
    repo_path = state.get("repo_path") or ""
    if not repo_path:
        files = files_in_patch(state.get("patch", ""))
        return GateResult(
            passed=bool(files), files=files,
            errors=[] if files else ["no unified diff headers found"],
        )
    s = get_settings()
    return check_patch(repo_path, state.get("patch", ""), s.gate_test_command, s.gate_test_timeout_s)


def run_reviewer(state: NexusState) -> NexusState:
    s = get_settings()
    gate = _gate(state)
    logger.info(f"[reviewer] {gate.summary()}")

    if gate.passed:
        try:
            review: ReviewOutput = llm.call(
                "reviewer",
                REVIEWER_SYSTEM,
                f"## Issue\nTitle: {state['issue_title']}\nBody: {state['issue_body'][:1200]}\n\n"
                f"## Patch\n{state.get('patch', '')}\n\n"
                f"## Root cause claimed by author\n{state.get('root_cause', '')}\n\n"
                f"## Deterministic checks\n{gate.summary()}\n\nReview:",
                schema=ReviewOutput,
            )
            score, feedback, issues = review.score, review.feedback, list(review.issues_found)
        except llm.LLMUnavailable as e:
            # Keep the gate-passing patch as a candidate; just don't call it verified.
            logger.warning("[reviewer] LLM review unavailable: %s", str(e)[:200])
            score, feedback, issues = 0.0, "LLM review unavailable (provider error)", []
            state = {**state, "error": f"reviewer: {str(e)[:300]}"}
    else:
        score = 0.0
        feedback = "The patch failed deterministic checks. Fix these first:\n" + "\n".join(gate.errors)
        issues = gate.errors

    passed = gate.passed and score >= s.review_pass_threshold
    logger.info(f"[reviewer] score={score:.2f} passed={passed}")

    round_no = state.get("reflection_count", 0)
    history = list(state.get("history", [])) + [Attempt(
        round=round_no,
        agent="engineer" if round_no == 0 else "reflector",
        gate_passed=gate.passed,
        gate_summary=gate.summary(),
        review_score=score,
        passed=passed,
    )]

    update: NexusState = {
        **state,
        "gate": gate.to_dict(),
        "review_score": score,
        "review_feedback": feedback,
        "review_issues": issues,
        "review_passed": passed,
        "history": history,
        "status": "done" if passed else "reflecting",
    }

    # Rank attempts by (gate passed, score). Ties keep the earlier, simpler patch.
    current = (gate.passed, score)
    best = (state.get("best_gate_passed", False), state.get("best_score", -1.0))
    if current > best:
        update.update(
            best_patch=state.get("patch", ""),
            best_score=score,
            best_gate_passed=gate.passed,
            best_explanation=state.get("patch_explanation", ""),
            best_files=state.get("files_modified", []),
        )
    return update
