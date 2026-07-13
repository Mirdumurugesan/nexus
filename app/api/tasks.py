"""
FastAPI routes for task management.
POST /tasks  → submit a GitHub issue for processing
GET  /tasks/{id} → poll task status + results
"""
import json
import uuid
import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.llm import estimate_cost_usd
from app.db.database import get_db, SessionLocal
from app.db.models import Task, TaskStatus
from app.auth.dependencies import get_current_user, require_engineer
from app.auth.models import User
from app.tools.github_parser import fetch_github_issue, parse_github_issue_url
from app.rag.chunker import chunk_repository
from app.rag.embedder import index_chunks

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1", tags=["tasks"])

# Serializes clone+index per repository so concurrent tasks on the same repo
# cannot delete each other's chunks mid-flight (single-process scope, which
# matches the BackgroundTasks execution model).
_repo_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


# ── Request / Response schemas ────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    github_issue_url: str
    use_hyde: bool = True


class PlanStep(BaseModel):
    id: str
    description: str
    file_hint: str
    status: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    current_step: str | None
    github_issue_url: str
    issue_title: str | None
    generated_patch: str | None
    patch_explanation: str | None
    relevant_files: list[str] | None
    confidence: float | None
    review_score: float | None
    review_passed: bool | None
    plan: list[dict] | None
    cost_usd: float | None
    error_message: str | None
    created_at: str
    completed_at: str | None


# ── Background task pipeline ──────────────────────────────────────────────────

def run_pipeline(task_id: str, use_hyde: bool):
    """
    Full NEXUS pipeline (Phase 1-4):
    1. Parse GitHub issue
    2. Clone + index repository (RAG)
    3. LangGraph multi-agent: Planner → Engineer → Reviewer → Reflector

    NOTE: deliberately a *sync* function. Starlette runs sync background tasks
    in a worker thread, so the blocking work here (git clone, chunking,
    embedding, DB commits) never blocks the API event loop. The async agent
    graph is driven via asyncio.run() inside this worker thread.

    Creates its own DB session — background tasks outlive the request session.
    """
    import tempfile
    import git
    from app.agents.graph import run_nexus_pipeline

    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        db.close()
        return

    def update_status(status: TaskStatus, step: str):
        task.status = status
        task.current_step = step
        task.updated_at = datetime.utcnow()
        db.commit()

    try:
        # ── Step 1: Parse issue ──────────────────────────────────────
        update_status(TaskStatus.CLONING, "Fetching GitHub issue")
        issue = fetch_github_issue(task.github_issue_url)

        task.repo_url = issue.repo_url
        task.repo_name = issue.repo_full_name
        task.issue_number = issue.issue_number
        task.issue_title = issue.issue_title
        task.issue_body = issue.issue_body
        db.commit()

        # ── Step 2: Clone + chunk + index (serialized per repo) ─────
        update_status(TaskStatus.INDEXING, "Cloning and indexing repository")

        with _repo_locks[issue.repo_full_name]:
            with tempfile.TemporaryDirectory() as tmpdir:
                git.Repo.clone_from(issue.repo_url, tmpdir, depth=1)
                chunks = chunk_repository(tmpdir)
                logger.info("[pipeline] Chunked %d code chunks", len(chunks))

                indexed = index_chunks(chunks, repo_name=issue.repo_full_name)
                logger.info("[pipeline] Indexed %d chunks into Weaviate", indexed)

        # ── Step 3: Multi-agent pipeline (LangGraph) ────────────────
        update_status(TaskStatus.RETRIEVING, "Planner Agent: decomposing issue")

        final_state = asyncio.run(run_nexus_pipeline(
            task_id=str(task.id),
            issue_title=issue.issue_title,
            issue_body=issue.issue_body,
            repo_name=issue.repo_full_name,
            repo_url=issue.repo_url,
            use_hyde=use_hyde,
        ))

        # ── Save results ─────────────────────────────────────────────
        task.generated_patch = final_state.get("patch", "")
        task.patch_explanation = final_state.get("patch_explanation", "")
        task.relevant_files = json.dumps(final_state.get("files_modified", []))

        meta = {
            "plan": final_state.get("plan", []),
            "review_score": final_state.get("review_score", 0.0),
            "review_passed": final_state.get("review_passed", False),
            "review_feedback": final_state.get("review_feedback", ""),
            "confidence": final_state.get("confidence", 0.0),
            "reflection_count": final_state.get("reflection_count", 0),
        }
        task.error_message = None  # clear any old error
        task.meta_json = json.dumps(meta)

        # Token accounting + cost estimate (agent LLM calls; the HyDE
        # mini-model call is not counted — negligible cost)
        task.prompt_tokens = final_state.get("prompt_tokens", 0)
        task.completion_tokens = final_state.get("completion_tokens", 0)
        task.estimated_cost_usd = estimate_cost_usd(
            settings.primary_llm, task.prompt_tokens, task.completion_tokens
        )

        task.status = TaskStatus.COMPLETED
        task.current_step = "Done"
        task.completed_at = datetime.utcnow()
        db.commit()

        logger.info(
            "[pipeline] Task %s COMPLETED | confidence=%.2f review=%.2f passed=%s "
            "reflections=%d tokens=%d cost=$%.4f",
            task_id, meta["confidence"], meta["review_score"], meta["review_passed"],
            meta["reflection_count"], task.prompt_tokens + task.completion_tokens,
            task.estimated_cost_usd,
        )

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.current_step = "Failed"
        task.updated_at = datetime.utcnow()
        db.commit()
        logger.exception("[pipeline] Task %s FAILED: %s", task_id, e)
    finally:
        db.close()


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    """Submit a GitHub issue for autonomous patch generation. Requires engineer role."""
    # Reject malformed URLs up front instead of queueing a doomed task
    try:
        parse_github_issue_url(request.github_issue_url)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid GitHub issue URL (expected https://github.com/<owner>/<repo>/issues/<n>)",
        )

    task = Task(
        github_issue_url=request.github_issue_url,
        status=TaskStatus.QUEUED,
        current_step="Queued",
        user_id=current_user.id,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(
        run_pipeline,
        task_id=str(task.id),
        use_hyde=request.use_hyde,
    )

    return _task_to_response(task)


def _visible_tasks(db: Session, user: User):
    """Admins see all tasks; other users see their own + webhook tasks (no owner)."""
    q = db.query(Task)
    if user.role != "admin":
        q = q.filter((Task.user_id == user.id) | (Task.user_id.is_(None)))
    return q


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll task status and retrieve results."""
    task = _visible_tasks(db, current_user).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent tasks (scoped to the requesting user unless admin)."""
    tasks = (
        _visible_tasks(db, current_user)
        .order_by(Task.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_task_to_response(t) for t in tasks]


@router.get("/health")
async def health():
    return {"status": "ok", "phase": "2-4", "version": "0.2.0"}


def _task_to_response(task: Task) -> TaskResponse:
    relevant_files = None
    plan = None
    review_score = None
    review_passed = None
    confidence = None

    if task.relevant_files:
        try:
            relevant_files = json.loads(task.relevant_files)
        except Exception:
            pass

    if task.meta_json:
        try:
            meta = json.loads(task.meta_json)
            plan = meta.get("plan")
            review_score = meta.get("review_score")
            review_passed = meta.get("review_passed")
            confidence = meta.get("confidence")
        except Exception:
            pass

    return TaskResponse(
        task_id=str(task.id),
        status=task.status.value,
        current_step=task.current_step,
        github_issue_url=task.github_issue_url,
        issue_title=task.issue_title,
        generated_patch=task.generated_patch,
        patch_explanation=task.patch_explanation,
        relevant_files=relevant_files,
        confidence=confidence,
        review_score=review_score,
        review_passed=review_passed,
        plan=plan,
        cost_usd=task.estimated_cost_usd,
        error_message=task.error_message,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )
