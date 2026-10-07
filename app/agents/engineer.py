"""
Engineer Agent — retrieves code context and writes the first patch.
"""
import logging
from pydantic import BaseModel, Field

from app.core import llm
from app.core.config import get_settings
from app.agents.state import NexusState
from app.rag.retriever import format_context_for_llm, hybrid_retrieve
from app.tools.edits import Edit, build_patch
from app.tools.patch_gate import files_in_patch

logger = logging.getLogger(__name__)

ENGINEER_SYSTEM = """You are an expert software engineer fixing a GitHub issue.

You get a plan and code context retrieved from the repository. Code context lines
are prefixed with their real line number as `  N | `. NEVER copy those prefixes
into the diff.

Return the fix as search/replace edits. For each edit, `search` must be lines
copied EXACTLY from the code context (same indentation, without the `  N | `
prefix) and should be just large enough to be unique; `replace` is what those
lines become. To create a new file use an empty `search`. Change only what the
fix needs and keep the existing style. Do not return a diff."""


class EngineerOutput(BaseModel):
    # Only `edits` is required: in the first live SWE-bench run, Groq rejected
    # otherwise-good tool calls for a missing `root_cause`.
    edits: list[Edit] = Field(description="Search/replace edits that implement the fix")
    root_cause: str = Field(default="", description="One sentence: the root cause of the bug")
    approach: str = Field(default="", description="One sentence: how the patch fixes it")
    confidence: float = Field(default=0.5, description="0.0-1.0 confidence in the fix", ge=0.0, le=1.0)
    test_hint: str = Field(default="", description="What to test to verify the fix")


def run_engineer(state: NexusState) -> NexusState:
    logger.info(f"[engineer] Retrieving context for: {state['issue_title']}")
    retrieved = hybrid_retrieve(
        issue_title=state["issue_title"],
        issue_body=state["issue_body"],
        repo_name=state.get("index_key") or state["repo_name"],
        top_k=12,
        use_hyde=state.get("use_hyde", True),
    )
    context = format_context_for_llm(retrieved, max_tokens=get_settings().context_tokens)
    retrieved_files = list(dict.fromkeys(c.file_path for c in retrieved))

    plan_text = "\n".join(
        f"{i + 1}. [{st['file_hint']}] {st['description']}" for i, st in enumerate(state.get("plan", []))
    )
    result: EngineerOutput = llm.call(
        "engineer",
        ENGINEER_SYSTEM,
        f"## GitHub Issue\nTitle: {state['issue_title']}\nBody: {state['issue_body'][:2000]}\n\n"
        f"## Plan\n{plan_text}\n\n## Relevant Code Context\n{context}\n\nReturn the edits:",
        schema=EngineerOutput,
    )

    patch, edit_errors = build_patch(state.get("repo_path", ""), result.edits)
    logger.info(f"[engineer] Patch drafted (self-reported confidence {result.confidence:.2f})")
    return {
        **state,
        "retrieved_context": context,
        "retrieved_files": retrieved_files,
        "patch": patch,
        "patch_explanation": (
            f"Root cause: {result.root_cause}\nApproach: {result.approach}\nTest: {result.test_hint}"
        ),
        "edits": [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in result.edits],
        "edit_errors": edit_errors,
        "files_modified": files_in_patch(patch) or sorted({(e.file if hasattr(e, "file") else e["file"]) for e in result.edits}),
        "confidence": result.confidence,
        "root_cause": result.root_cause,
        "status": "reviewing",
    }
