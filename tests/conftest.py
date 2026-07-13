"""
Pytest configuration — sets up an in-memory SQLite DB for tests
so tests don't hit the real Supabase instance, and overrides
authentication so API tests run as a fake engineer user.
"""
import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base, get_db
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test.db"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Fake authenticated user for API tests. require_engineer / require_admin
# resolve through get_current_user, so this single override covers them.
TEST_USER = User(
    id=uuid.uuid4(),
    email="test@nexus.local",
    username="testuser",
    hashed_password="not-a-real-hash",
    full_name="Test User",
    role="engineer",
    is_active=True,
    created_at=datetime.utcnow(),
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_get_current_user():
    return TEST_USER


@pytest.fixture(autouse=True, scope="session")
def setup_test_db():
    """Create tables in test DB once per session."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()
