"""
FastAPI routes for task management.
POST /tasks  → submit a GitHub issue for processing
GET  /tasks/{id} → poll task status + results
"""
import logging
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator

from app.db.database import get_db, SessionLocal
from app.db.models import Task, TaskStatus
from app.auth.dependencies import get_current_user, require_engineer
from app.auth.models import User
from app.tools.github_parser import fetch_github_issue, parse_github_issue_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tasks"])


# ── Request / Response schemas ────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    github_issue_url: str
    use_hyde: bool = True

    @field_validator("github_issue_url")
    @classmethod
    def must_be_issue_url(cls, v: str) -> str:
        parse_github_issue_url(v)  # raises ValueError -> 422
        return v.strip()


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
    gate_passed: bool | None = None
    reflections: int | None = None
    history: list[dict] | None = None
    plan: list[dict] | None
    cost_usd: float | None
    error_message: str | None
    created_at: str
    completed_at: str | None


# ── Background task pipeline ──────────────────────────────────────────────────

def _parse_task_id(task_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(task_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Task not found")


def run_pipeline(task_id: str, use_hyde: bool = True):
    """
    Background worker: fetch issue -> solve_issue() -> persist result.
    Sync on purpose: FastAPI runs sync background tasks in a thread pool, so a
    long agent run never blocks the event loop. Uses its own DB session.
    """
    from app.pipeline import solve_issue

    db = SessionLocal()
    task = db.query(Task).filter(Task.id == _parse_task_id(task_id)).first()
    if not task:
        db.close()
        return

    def step(status: TaskStatus, text: str):
        task.status = status
        task.current_step = text
        task.updated_at = datetime.utcnow()
        db.commit()

    try:
        step(TaskStatus.CLONING, "Fetching GitHub issue")
        issue = fetch_github_issue(task.github_issue_url)
        task.repo_url = issue.repo_url
        task.repo_name = issue.repo_full_name
        task.issue_number = issue.issue_number
        task.issue_title = issue.issue_title
        task.issue_body = issue.issue_body
        db.commit()

        step(TaskStatus.GENERATING, "Indexing repo + running Planner → Engineer → Reviewer ⇄ Reflector")
        result = solve_issue(
            issue_title=issue.issue_title,
            issue_body=issue.issue_body,
            repo_name=issue.repo_full_name,
            repo_url=issue.repo_url,
            use_hyde=use_hyde,
            task_id=str(task.id),
        )

        task.generated_patch = result.patch
        task.patch_explanation = result.patch_explanation
        task.relevant_files = json.dumps(result.files_modified)
        task.meta_json = json.dumps({
            "plan": result.plan,
            "review_score": result.review_score,
            "review_passed": result.passed,
            "gate_passed": result.gate_passed,
            "gate": result.gate,
            "confidence": result.confidence,
            "reflection_count": result.reflections,
            "history": result.history,
            "retrieved_files": result.retrieved_files,
            "llm_calls": result.llm_calls,
            "seconds": result.seconds,
            "budget_exhausted": result.budget_exhausted,
        })
        task.prompt_tokens = result.prompt_tokens
        task.completion_tokens = result.completion_tokens
        task.estimated_cost_usd = result.cost_usd
        task.error_message = None
        task.status = TaskStatus.COMPLETED
        task.current_step = "Done — patch verified" if result.passed else "Done — best attempt (not verified)"
        task.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"[pipeline] {task_id} done: passed={result.passed} gate={result.gate_passed} "
              f"score={result.review_score:.2f} reflections={result.reflections} {result.seconds}s")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        task.status = TaskStatus.FAILED
        task.error_message = f"{type(e).__name__}: {e}"[:2000]
        task.current_step = "Failed"
        task.updated_at = datetime.utcnow()
        db.commit()
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
    """Admins see every task; others see their own plus webhook tasks (no owner)."""
    q = db.query(Task)
    if user.role != "admin":
        q = q.filter((Task.user_id == user.id) | (Task.user_id.is_(None)))
    return q


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll task status and retrieve results."""
    task = _visible_tasks(db, current_user).filter(Task.id == _parse_task_id(task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent tasks."""
    tasks = _visible_tasks(db, current_user).order_by(Task.created_at.desc()).limit(limit).all()
    return [_task_to_response(t) for t in tasks]


@router.get("/health")
async def health():
    from app.core import llm
    from app.core.config import get_settings
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.3.0",
        "llm_configured": llm.available(),
        "retriever": s.retriever,
        "gate_tests": bool(s.gate_test_command),
    }


def _task_to_response(task: Task) -> TaskResponse:
    relevant_files = None
    plan = None
    review_score = None
    review_passed = None
    confidence = None
    meta: dict = {}

    if task.relevant_files:
        try:
            relevant_files = json.loads(task.relevant_files)
        except Exception:
            pass

    if hasattr(task, "meta_json") and task.meta_json:
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
        gate_passed=meta.get("gate_passed"),
        reflections=meta.get("reflection_count"),
        history=meta.get("history"),
        plan=plan,
        cost_usd=task.estimated_cost_usd,
        error_message=task.error_message,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )
