"""
GitHub App Webhook Handler
──────────────────────────
Receives GitHub webhook events and auto-triggers NEXUS when an issue is
labeled with one of the trigger labels ("nexus", "auto-fix", ...).

SECURITY: the endpoint is DISABLED (503) unless GITHUB_WEBHOOK_SECRET is set,
and every request must carry a valid HMAC-SHA256 signature. Auto-triggering
on every opened issue was removed deliberately — each run costs real LLM
money, so triggering requires an explicit label from a repo collaborator.

Setup:
1. Go to GitHub → Settings → Developer settings → GitHub Apps → New
2. Set webhook URL to: https://your-domain.com/api/v1/webhook/github
3. Subscribe to "Issues" events
4. Set a webhook secret and add GITHUB_WEBHOOK_SECRET to your .env
"""
import hashlib
import hmac
import json
import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from app.core.config import get_settings
from app.api.tasks import run_pipeline
from app.db.database import SessionLocal
from app.db.models import Task, TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])

# Labels that trigger NEXUS auto-fix
TRIGGER_LABELS = {"nexus", "auto-fix", "nexus-fix", "ai-fix"}


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive GitHub webhook events.
    Triggers the NEXUS pipeline when a trigger label is added to an issue.
    """
    # Signature verification is mandatory — no secret, no webhook.
    settings = get_settings()
    webhook_secret = settings.github_webhook_secret
    if not webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Webhook disabled: GITHUB_WEBHOOK_SECRET is not configured",
        )

    payload_bytes = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    event_type = request.headers.get("X-GitHub-Event", "")

    if not verify_github_signature(payload_bytes, signature, webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Only handle issue events
    if event_type != "issues":
        return {"status": "ignored", "reason": f"event={event_type}"}

    try:
        payload = json.loads(payload_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    action = payload.get("action", "")
    issue = payload.get("issue", {})
    repo = payload.get("repository", {})

    issue_url = issue.get("html_url", "")
    issue_number = issue.get("number")
    issue_title = issue.get("title", "")
    repo_full_name = repo.get("full_name", "")

    # Trigger ONLY when a trigger label is added (explicit, cost-bounded opt-in)
    should_trigger = False
    trigger_reason = ""

    if action == "labeled":
        label_name = payload.get("label", {}).get("name", "").lower()
        if label_name in TRIGGER_LABELS:
            should_trigger = True
            trigger_reason = f"label '{label_name}' added"

    if not should_trigger or not issue_url:
        return {"status": "ignored", "action": action}

    # Create task in DB (no owner — webhook tasks are system-initiated)
    db = SessionLocal()
    try:
        task = Task(
            github_issue_url=issue_url,
            status=TaskStatus.QUEUED,
            current_step="Queued via webhook",
            issue_title=issue_title,
            repo_name=repo_full_name,
            issue_number=issue_number,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        task_id = str(task.id)
    finally:
        db.close()

    # Trigger pipeline in background
    background_tasks.add_task(run_pipeline, task_id=task_id, use_hyde=True)

    logger.info(
        "[webhook] Auto-triggered NEXUS for %s#%s (%s) → task %s",
        repo_full_name, issue_number, trigger_reason, task_id,
    )

    return {
        "status": "triggered",
        "task_id": task_id,
        "issue_url": issue_url,
        "trigger_reason": trigger_reason,
    }


@router.get("/github/health")
async def webhook_health():
    return {
        "status": "ok",
        "configured": bool(get_settings().github_webhook_secret),
        "trigger_labels": sorted(TRIGGER_LABELS),
        "auto_trigger_on_open": False,
    }
