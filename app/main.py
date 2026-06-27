from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from sqlalchemy import text
from app.api.tasks import router
from app.api.webhook import router as webhook_router
from app.api.metrics import router as metrics_router
from app.auth.router import router as auth_router
from app.db.database import engine
from app.db import models
from app.auth import models as auth_models  # ensure User table is created
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables (tasks + users)
    models.Base.metadata.create_all(bind=engine)
    auth_models.Base.metadata.create_all(bind=engine)
    # Safe migrations
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS meta_json TEXT"))
        conn.commit()
    print("[NEXUS] Database tables created/verified.")
    yield
    print("[NEXUS] Shutting down.")


app = FastAPI(
    title="NEXUS",
    description="Multi-Agent Autonomous Software Engineering Platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
