import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.api.tasks import router
from app.api.webhook import router as webhook_router
from app.api.metrics import router as metrics_router
from app.auth.router import router as auth_router
from app.db.database import engine, SessionLocal
from app.db import models
from app.db.models import Task, TaskStatus
from app.auth import models as auth_models  # ensure User table is created

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nexus")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables (tasks + users)
    models.Base.metadata.create_all(bind=engine)
    auth_models.Base.metadata.create_all(bind=engine)

    # Safe additive migrations (Postgres only; SQLite is used in tests and
    # does not support ADD COLUMN IF NOT EXISTS)
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS meta_json TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS user_id UUID"))
            conn.commit()

    # Orphan sweep: BackgroundTasks run in-process, so any task still in a
    # non-terminal state at boot died with the previous process. Mark it
    # failed instead of leaving it stuck forever. (Assumes a single uvicorn
    # worker, which is the documented deployment model for v1.)
    with SessionLocal() as db:
        orphaned = (
            db.query(Task)
            .filter(Task.status.notin_([TaskStatus.COMPLETED, TaskStatus.FAILED]))
            .update(
                {
                    Task.status: TaskStatus.FAILED,
                    Task.current_step: "Failed",
                    Task.error_message: "Orphaned by server restart",
                },
                synchronize_session=False,
            )
        )
        db.commit()
        if orphaned:
            logger.warning("Marked %d orphaned in-flight task(s) as failed.", orphaned)

    logger.info("[NEXUS] Database tables created/verified.")
    yield
    logger.info("[NEXUS] Shutting down.")


app = FastAPI(
    title="NEXUS",
    description="Multi-Agent Autonomous Software Engineering Platform",
    version="0.2.0",
    lifespan=lifespan,
)

# Explicit origin allowlist (wildcard + credentials is invalid per the CORS
# spec and unsafe). The dashboard is served by this app, so it is same-origin
# and does not rely on CORS at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routes (public — login/register)
app.include_router(auth_router)

# Protected routes
app.include_router(router)
app.include_router(webhook_router)
app.include_router(metrics_router)


@app.get("/")
async def root():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"name": "NEXUS", "docs": "/docs"}


@app.get("/login.html")
async def login_page():
    login_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "login.html")
    return FileResponse(login_path)


@app.get("/index.html")
async def dashboard_page():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path)
