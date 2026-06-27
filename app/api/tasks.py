"""
FastAPI routes for task management.
POST /tasks  → submit a GitHub issue for processing
GET  /tasks/{id} → poll task status + results
"""
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db, SessionLocal
from app.db.models import Task, TaskStatus
from app.auth.dependencies import get_current_user, require_engineer
from app.auth.models import User
from app.tools.github_parser import fetch_github_issue
from app.rag.chunker import chunk_repository
from app.rag.embedder import index_chunks

router = APIRouter(prefix="/api/v1", tags=["tasks"])


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

async def run_pipeline(task_id: str, use_hyde: bool):
    """
    Full NEXUS pipeline (Phase 1-4):
    1. Parse GitHub issue
    2. Clone + index repository (RAG)
    3. LangGraph multi-agent: Planner → Engineer → Reviewer → Reflector

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

        # ── Step 2: Clone + chunk + index ───────────────────────────
        update_status(TaskStatus.INDEXING, "Cloning and indexing repository")

        with tempfile.TemporaryDirectory() as tmpdir:
            git.Repo.clone_from(issue.repo_url, tmpdir, depth=1)
            chunks = chunk_repository(tmpdir)
            print(f"[pipeline] Chunked {len(chunks)} code chunks")

            indexed = index_chunks(chunks, repo_name=issue.repo_full_name)
            print(f"[pipeline] Indexed {indexed} chunks into Weaviate")

        # ── Step 3: Multi-agent pipeline (LangGraph) ────────────────
        update_status(TaskStatus.RETRIEVING, "Planner Agent: decomposing issue")

        final_state = await run_nexus_pipeline(
            task_id=str(task.id),
            issue_title=issue.issue_title,
            issue_body=issue.issue_body,
            repo_name=issue.repo_full_name,
            repo_url=issue.repo_url,
        )

        # ── Save results ─────────────────────────────────────────────
        task.generated_patch = final_state.get("patch", "")
        task.patch_explanation = final_state.get("patch_explanation", "")
        task.relevant_files = json.dumps(final_state.get("files_modified", []))

        # Store plan + review scores in error_message field (reuse for now)
        meta = {
            "plan": final_state.get("plan", []),
            "review_score": final_state.get("review_score", 0.0),
            "review_passed": final_state.get("review_passed", False),
            "review_feedback": final_state.get("review_feedback", ""),
            "confidence": final_state.get("confidence", 0.0),
            "reflection_count": final_state.get("reflection_count", 0),
        }
        task.error_message = None  # clear any old error
        # Store meta in a new JSON column — we'll add it to the model
        task.meta_json = json.dumps(meta)

        task.status = TaskStatus.COMPLETED
        task.current_step = "Done"
        task.completed_at = datetime.utcnow()
        db.commit()

        print(f"[pipeline] Task {task_id} COMPLETED ✓")
        print(f"  Confidence: {meta['confidence']:.2f}")
        print(f"  Review score: {meta['review_score']:.2f} | Passed: {meta['review_passed']}")
        print(f"  Reflections: {meta['reflection_count']}")

    except Exception as e:
        import traceback
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.current_step = "Failed"
        task.updated_at = datetime.utcnow()
        db.commit()
        print(f"[pipeline] Task {task_id} FAILED: {e}")
        traceback.print_exc()
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


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll task status and retrieve results."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent tasks."""
    tasks = db.query(Task).order_by(Task.created_at.desc()).limit(limit).all()
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
        plan=plan,
        cost_usd=task.estimated_cost_usd,
        error_message=task.error_message,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )
