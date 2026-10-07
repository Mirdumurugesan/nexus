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
from app.tools.edits import Edit, build_patch
from app.tools.patch_gate import files_in_patch

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM = """You are an expert software engineer repairing a rejected patch.

Address EVERY problem in the feedback. Return the COMPLETE set of search/replace
edits against the ORIGINAL files (earlier edits were not kept). If a search block
was "not found", copy the lines exactly from the code context (same indentation,
without the `  N | ` prefix). Keep edits minimal. Do not return a diff."""

# Kept for backwards compatibility with imports elsewhere.
MAX_REFLECTIONS = get_settings().max_reflections


class ReflectorOutput(BaseModel):
    edits: list[Edit] = Field(description="Complete search/replace edits against the original files")
    changes_made: str = Field(description="What changed vs the previous patch and why")
    new_confidence: float = Field(description="Confidence in the improved patch 0.0-1.0", ge=0.0, le=1.0)


def _fmt_edits(edits: list) -> str:
    if not edits:
        return "(none)"
    return "\n".join(
        f"--- edit {i + 1}: {e.get('file')}\nSEARCH:\n{e.get('search', '')}\nREPLACE:\n{e.get('replace', '')}"
        for i, e in enumerate(edits)
    )


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
        f"## Rejected edits\n{_fmt_edits(state.get('edits', []))}\n\n"
        f"## Resulting diff\n{state.get('patch', '') or '(none: the edits could not be applied)'}\n\n"
        f"## Review (score {state.get('review_score', 0):.2f})\n{state.get('review_feedback', '')}\n"
        f"Issues:\n{issues}\n\n"
        f"## Code context\n{state.get('retrieved_context', '')[:get_settings().context_tokens * 2]}\n\nReturn the improved edits:",
        schema=ReflectorOutput,
    )

    patch, edit_errors = build_patch(state.get("repo_path", ""), result.edits)
    return {
        **state,
        "patch": patch,
        "edits": [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in result.edits],
        "edit_errors": edit_errors,
        "files_modified": files_in_patch(patch) or state.get("files_modified", []),
        "patch_explanation": state.get("patch_explanation", "")
        + f"\n[Reflection {round_no}]: {result.changes_made}",
        "confidence": result.new_confidence,
        "reflection_count": round_no,
        "status": "reviewing",
    }
