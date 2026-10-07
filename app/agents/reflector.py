"""
Reflector Agent — rewrites a rejected patch using the reviewer's feedback.

The feedback is usually a *fact* from the Patch Gate ("patch does not apply at
pricing.py:14", "SyntaxError line 22", a failing test) rather than an opinion,
which is what makes the loop converge instead of wander.
"""
import logging
from pydantic import BaseModel, Field

from app.core import llm
from app.agents.state import NexusState
from app.core.config import get_settings
from app.tools.patch_gate import clean_patch, files_in_patch

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM = """You are an expert software engineer repairing a rejected patch.

Address EVERY problem in the feedback. If the patch did not apply, the context
lines did not match the file: re-read the code context (lines are prefixed with
`  N | `, never copy the prefix) and reproduce context lines exactly.
Return a complete, minimal unified diff."""

# Kept for backwards compatibility with imports elsewhere.
MAX_REFLECTIONS = get_settings().max_reflections


class ReflectorOutput(BaseModel):
    improved_patch: str = Field(description="The improved patch in unified diff format")
    changes_made: str = Field(description="What changed vs the previous patch and why")
    new_confidence: float = Field(description="Confidence in the improved patch 0.0-1.0", ge=0.0, le=1.0)


def run_reflector(state: NexusState) -> NexusState:
    max_r = get_settings().max_reflections
    round_no = state.get("reflection_count", 0) + 1
    logger.info(f"[reflector] round {round_no}/{max_r}")

    if round_no > max_r:
        return {**state, "reflection_count": round_no, "status": "done"}

    issues = "\n".join(f"- {i}" for i in state.get("review_issues", [])) or "- (none listed)"
    result: ReflectorOutput = llm.call(
        "reflector",
        REFLECTOR_SYSTEM,
        f"## Issue\nTitle: {state['issue_title']}\nBody: {state['issue_body'][:1200]}\n\n"
        f"## Rejected patch\n{state.get('patch', '')}\n\n"
        f"## Review (score {state.get('review_score', 0):.2f})\n{state.get('review_feedback', '')}\n"
        f"Issues:\n{issues}\n\n"
        f"## Code context\n{state.get('retrieved_context', '')[:12000]}\n\nImproved diff:",
        schema=ReflectorOutput,
    )

    patch = clean_patch(result.improved_patch)
    return {
        **state,
        "patch": patch,
        "files_modified": files_in_patch(patch) or state.get("files_modified", []),
        "patch_explanation": state.get("patch_explanation", "")
        + f"\n[Reflection {round_no}]: {result.changes_made}",
        "confidence": result.new_confidence,
        "reflection_count": round_no,
        "status": "reviewing",
    }
