"""
solve_issue() — the whole NEXUS run as one function, shared by the API worker,
the CLI and the SWE-bench evaluator so all three exercise identical code.

    checkout (clone, optionally pinned to a commit)
      -> AST chunking -> index (local BM25 or Weaviate)
      -> LangGraph: planner -> engineer -> reviewer <-> reflector -> finalize
"""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field

from app.core import llm
from app.core.config import get_settings
from app.agents.graph import run_graph
from app.rag.chunker import chunk_repository
from app.rag.retriever import index_repository

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    patch: str
    passed: bool
    gate_passed: bool
    review_score: float
    confidence: float
    files_modified: list[str]
    retrieved_files: list[str]
    plan: list[dict]
    patch_explanation: str
    reflections: int
    history: list[dict]
    gate: dict
    chunks_indexed: int
    llm_calls: int
    llm_failovers: list[str]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    budget_exhausted: bool
    seconds: float
    error: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def checkout(repo_url: str, dest: str, base_commit: str | None = None) -> str:
    """Clone `repo_url` into `dest`. With base_commit, pin to it (needed for SWE-bench)."""
    if base_commit:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", repo_url, dest], check=True)
        subprocess.run(["git", "checkout", "-q", base_commit], cwd=dest, check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", repo_url, dest], check=True)
    return dest


def solve_issue(
    issue_title: str,
    issue_body: str,
    repo_name: str,
    repo_url: str = "",
    repo_path: str | None = None,
    base_commit: str | None = None,
    use_hyde: bool = True,
    task_id: str = "",
) -> SolveResult:
    """
    Run NEXUS end to end. Pass `repo_path` to use an existing checkout (left
    untouched afterwards: the patch gate always restores it); otherwise the
    repo is cloned to a temp dir that is deleted when done.
    """
    t0 = time.time()
    tmp = None
    # Unique per run so concurrent tasks on the same repo never share an index.
    index_key = f"{repo_name}#{uuid.uuid4().hex[:8]}"
    if not repo_path:
        tmp = tempfile.mkdtemp(prefix="nexus-")
        repo_path = checkout(repo_url, os.path.join(tmp, "repo"), base_commit)

    try:
        chunks = chunk_repository(repo_path)
        indexed = index_repository(chunks, index_key)
        logger.info(f"[pipeline] indexed {indexed} chunks from {repo_name}")

        with llm.track_usage() as usage:
            state = run_graph(
                task_id=task_id,
                issue_title=issue_title,
                issue_body=issue_body,
                repo_name=repo_name,
                repo_url=repo_url,
                repo_path=repo_path,
                index_key=index_key,
                use_hyde=use_hyde,
            )

        gate = state.get("gate", {})
        return SolveResult(
            patch=state.get("patch", ""),
            passed=bool(state.get("review_passed")),
            gate_passed=bool(state.get("best_gate_passed") or gate.get("passed")),
            review_score=float(state.get("review_score", 0.0)),
            confidence=float(state.get("confidence", 0.0)),
            files_modified=list(state.get("files_modified", [])),
            retrieved_files=list(state.get("retrieved_files", [])),
            plan=[dict(p) for p in state.get("plan", [])],
            patch_explanation=state.get("patch_explanation", ""),
            reflections=int(state.get("reflection_count", 0)),
            history=[dict(h) for h in state.get("history", [])],
            gate=gate,
            chunks_indexed=indexed,
            llm_calls=usage.calls,
            llm_failovers=usage.failures,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=round(usage.cost_usd, 6),
            budget_exhausted=usage.total_tokens >= get_settings().max_tokens_per_task,
            seconds=round(time.time() - t0, 2),
            error=state.get("error", ""),
        )
    finally:
        from app.rag.local_index import drop_local_index
        drop_local_index(index_key)
        if get_settings().retriever == "weaviate":
            try:
                from app.rag.embedder import delete_repo_chunks
                delete_repo_chunks(index_key)  # per-run index: no cross-task clobbering
            except Exception as e:  # noqa: BLE001
                logger.warning("[pipeline] Weaviate cleanup failed: %s", e)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
