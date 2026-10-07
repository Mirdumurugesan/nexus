"""
Test setup: SQLite, no LLM keys, local retriever. Must run before `app` imports
because settings are read at import time.
"""
import os
import subprocess
import tempfile

_tmp = tempfile.mkdtemp(prefix="nexus-test-")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_tmp}/test.db",
    "OPENAI_API_KEY": "",
    "GROQ_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "GITHUB_TOKEN": "",
    "GITHUB_WEBHOOK_SECRET": "",
    "RETRIEVER": "local",
    "GATE_TEST_COMMAND": "",
    "APP_ENV": "test",
    "SECRET_KEY": "test-secret",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def make_user(client, username: str) -> dict:
    r = client.post("/api/v1/auth/register", json={
        "email": f"{username}@nexus.dev", "username": username, "password": "s3cret-pass",
    })
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    """The first account registered becomes admin."""
    return make_user(client, "admin1")


@pytest.fixture(scope="session")
def auth(client, admin):
    """A regular engineer."""
    return make_user(client, "engineer1")


@pytest.fixture(scope="session")
def other(client, admin):
    return make_user(client, "engineer2")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def wealth_repo(tmp_path):
    """A committed copy of the demo fixture (real bug, real tests)."""
    from app.demo import _make_repo
    return _make_repo(str(tmp_path / "wealth-lib"))
