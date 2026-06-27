"""
GitHub App Webhook Handler
──────────────────────────
Receives GitHub webhook events and auto-triggers NEXUS when:
- A new issue is opened (action: "opened")
- An issue is labeled with "nexus" or "auto-fix"

Setup:
1. Go to GitHub → Settings → Developer settings → GitHub Apps → New
2. Set webhook URL to: https://your-domain.com/api/v1/webhook/github
3. Subscribe to "Issues" events
4. Add GITHUB_WEBHOOK_SECRET to your .env
"""
import hashlib
import hmac
import json
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from app.core.config import get_settings
from app.api.tasks import run_pipeline
from app.db.database import SessionLocal
from app.db.models import Task, TaskStatus

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])
settings = get_settings()

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
    Triggers NEXUS pipeline when an issue is opened or labeled.
    """
    payload_bytes = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    event_type = request.headers.get("X-GitHub-Event", "")

    # Verify signature if secret is configured
    webhook_secret = getattr(settings, "github_webhook_secret", "")
    if webhook_secret:
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

    # Trigger on: new issue opened, OR label "nexus"/"auto-fix" added
    should_trigger = False
    trigger_reason = ""

    if action == "opened":
        should_trigger = True
        trigger_reason = "issue opened"

    elif action == "labeled":
        label_name = payload.get("label", {}).get("name", "").lower()
        if label_name in TRIGGER_LABELS:
            should_trigger = True
            trigger_reason = f"label '{label_name}' added"

    if not should_trigger or not issue_url:
        return {"status": "ignored", "action": action}

    # Create task in DB
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

    print(f"[webhook] Auto-triggered NEXUS for {repo_full_name}#{issue_number} ({trigger_reason})")
    print(f"[webhook] Task ID: {task_id}")

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
        "trigger_labels": list(TRIGGER_LABELS),
        "auto_trigger_on_open": True,
    }
