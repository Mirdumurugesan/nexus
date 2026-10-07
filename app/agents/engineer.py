"""
Engineer Agent — retrieves code context and writes the first patch.
"""
import logging
from pydantic import BaseModel, Field

from app.core import llm
from app.agents.state import NexusState
from app.rag.retriever import format_context_for_llm, hybrid_retrieve
from app.tools.patch_gate import clean_patch, files_in_patch

logger = logging.getLogger(__name__)

ENGINEER_SYSTEM = """You are an expert software engineer fixing a GitHub issue.

You get a plan and code context retrieved from the repository. Code context lines
are prefixed with their real line number as `  N | `. NEVER copy those prefixes
into the diff.

Return a minimal unified diff (--- a/path, +++ b/path, @@ hunks) that applies
with `git apply`. Context lines must match the file exactly, including
indentation. Change only what the fix needs; keep the existing style."""


class EngineerOutput(BaseModel):
    root_cause: str = Field(description="One sentence: the root cause of the bug")
    approach: str = Field(description="One sentence: how the patch fixes it")
    patch: str = Field(description="Complete unified diff (--- a/file +++ b/file)")
    files_modified: list[str] = Field(default_factory=list, description="File paths modified")
    confidence: float = Field(description="0.0-1.0 confidence in the fix", ge=0.0, le=1.0)
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
    context = format_context_for_llm(retrieved, max_tokens=6000)
    retrieved_files = list(dict.fromkeys(c.file_path for c in retrieved))

    plan_text = "\n".join(
        f"{i + 1}. [{st['file_hint']}] {st['description']}" for i, st in enumerate(state.get("plan", []))
    )
    result: EngineerOutput = llm.call(
        "engineer",
        ENGINEER_SYSTEM,
        f"## GitHub Issue\nTitle: {state['issue_title']}\nBody: {state['issue_body'][:2000]}\n\n"
        f"## Plan\n{plan_text}\n\n## Relevant Code Context\n{context}\n\nWrite the unified diff:",
        schema=EngineerOutput,
    )

    patch = clean_patch(result.patch)
    logger.info(f"[engineer] Patch drafted (self-reported confidence {result.confidence:.2f})")
    return {
        **state,
        "retrieved_context": context,
        "retrieved_files": retrieved_files,
        "patch": patch,
        "patch_explanation": (
            f"Root cause: {result.root_cause}\nApproach: {result.approach}\nTest: {result.test_hint}"
        ),
        "files_modified": files_in_patch(patch) or result.files_modified,
        "confidence": result.confidence,
        "root_cause": result.root_cause,
        "status": "reviewing",
    }
