import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Integer, Text, Enum
from sqlalchemy.dialects.postgresql import UUID
import enum
from app.db.database import Base


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    CLONING = "cloning"
    INDEXING = "indexing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    github_issue_url = Column(String(500), nullable=False)
    repo_url = Column(String(500), nullable=True)
    repo_name = Column(String(200), nullable=True)
    issue_number = Column(Integer, nullable=True)
    issue_title = Column(String(500), nullable=True)
    issue_body = Column(Text, nullable=True)

    status = Column(Enum(TaskStatus), default=TaskStatus.QUEUED, nullable=False)
    current_step = Column(String(100), nullable=True)

    generated_patch = Column(Text, nullable=True)
    patch_explanation = Column(Text, nullable=True)
    relevant_files = Column(Text, nullable=True)  # JSON string

    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)

    error_message = Column(Text, nullable=True)
    meta_json = Column(Text, nullable=True)   # stores plan, review scores, reflection count

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)