"""
Metrics & Analytics API
────────────────────────
Provides aggregate statistics across all tasks.
Useful for the dashboard and for proving system performance in interviews.

GET /api/v1/metrics         → overall stats
GET /api/v1/metrics/daily   → daily breakdown
"""
import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import get_db
from app.db.models import Task, TaskStatus

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("")
async def get_metrics(db: Session = Depends(get_db)):
    """Overall platform metrics."""
    total = db.query(Task).count()
    completed = db.query(Task).filter(Task.status == TaskStatus.COMPLETED).count()
    failed = db.query(Task).filter(Task.status == TaskStatus.FAILED).count()
    in_progress = db.query(Task).filter(
        Task.status.in_([
            TaskStatus.QUEUED, TaskStatus.CLONING,
            TaskStatus.INDEXING, TaskStatus.RETRIEVING,
            TaskStatus.GENERATING
        ])
    ).count()

    success_rate = round(completed / total * 100, 1) if total else 0

    # Average confidence and review score from meta_json
    completed_tasks = db.query(Task).filter(
        Task.status == TaskStatus.COMPLETED,
        Task.meta_json.isnot(None),
    ).all()

    confidences = []
    review_scores = []
    review_passed_count = 0
    total_cost = 0.0

    for t in completed_tasks:
        try:
            meta = json.loads(t.meta_json)
            if meta.get("confidence"):
                confidences.append(meta["confidence"])
            if meta.get("review_score"):
                review_scores.append(meta["review_score"])
            if meta.get("review_passed"):
                review_passed_count += 1
        except Exception:
            pass
        if t.estimated_cost_usd:
            total_cost += t.estimated_cost_usd

    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
    avg_review = round(sum(review_scores) / len(review_scores), 3) if review_scores else None

    # Average completion time
    completed_with_time = db.query(Task).filter(
        Task.status == TaskStatus.COMPLETED,
        Task.completed_at.isnot(None),
    ).all()

    durations = [
        (t.completed_at - t.created_at).total_seconds()
        for t in completed_with_time
        if t.completed_at and t.created_at
    ]
    avg_duration_sec = round(sum(durations) / len(durations), 1) if durations else None

    # Most active repos
    repo_counts = {}
    for t in db.query(Task).filter(Task.repo_name.isnot(None)).all():
        repo_counts[t.repo_name] = repo_counts.get(t.repo_name, 0) + 1
    top_repos = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "summary": {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "success_rate_pct": success_rate,
        },
        "quality": {
            "avg_confidence": avg_confidence,
            "avg_review_score": avg_review,
            "reviewer_passed_count": review_passed_count,
            "reviewer_passed_pct": round(review_passed_count / completed * 100, 1) if completed else 0,
        },
        "performance": {
            "avg_completion_time_sec": avg_duration_sec,
            "total_estimated_cost_usd": round(total_cost, 4),
        },
        "top_repos": [{"repo": r, "tasks": c} for r, c in top_repos],
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/daily")
async def get_daily_metrics(days: int = 7, db: Session = Depends(get_db)):
    """Task counts by day for the last N days."""
    since = datetime.utcnow() - timedelta(days=days)
    tasks = db.query(Task).filter(Task.created_at >= since).all()

    daily: dict[str, dict] = {}
    for t in tasks:
        day = t.created_at.strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = {"date": day, "total": 0, "completed": 0, "failed": 0}
        daily[day]["total"] += 1
        if t.status == TaskStatus.COMPLETED:
            daily[day]["completed"] += 1
        elif t.status == TaskStatus.FAILED:
            daily[day]["failed"] += 1

    return {
        "days": days,
        "data": sorted(daily.values(), key=lambda x: x["date"]),
    }
