import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.api.metrics import router as metrics_router
from app.api.tasks import router as tasks_router
from app.api.webhook import router as webhook_router
from app.auth import models as auth_models  # noqa: F401 — registers the users table
from app.auth.router import router as auth_router
from app.core.config import get_settings
from app.db import models
from app.db.database import SessionLocal, engine
from app.db.models import Task, TaskStatus

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nexus")

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(bind=engine)

    # Additive migrations for older Supabase databases (Postgres only).
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS meta_json TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS user_id UUID"))
            conn.commit()

    # Orphan sweep: agent runs are in-process, so anything still in flight at
    # boot died with the previous process. Mark it failed instead of stuck.
    # (Single uvicorn worker is the documented deployment model.)
    with SessionLocal() as db:
        orphaned = (
            db.query(Task)
            .filter(Task.status.notin_([TaskStatus.COMPLETED, TaskStatus.FAILED]))
            .update({Task.status: TaskStatus.FAILED, Task.current_step: "Failed",
                     Task.error_message: "Orphaned by server restart"}, synchronize_session=False)
        )
        db.commit()
        if orphaned:
            logger.warning("Marked %d orphaned in-flight task(s) as failed.", orphaned)

    logger.info("[NEXUS] Database ready.")
    yield
    logger.info("[NEXUS] Shutting down.")


app = FastAPI(
    title="NEXUS",
    description="Multi-agent issue-to-patch system with a deterministic patch gate",
    version="0.3.0",
    lifespan=lifespan,
)

# Explicit allowlist (wildcard + credentials is invalid per the CORS spec).
# The dashboard is served by this app, so it is same-origin anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(tasks_router)
app.include_router(webhook_router)
app.include_router(metrics_router)


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def dashboard():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


@app.get("/login.html", include_in_schema=False)
async def login_page():
    return FileResponse(os.path.join(FRONTEND, "login.html"))
