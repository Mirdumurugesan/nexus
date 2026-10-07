"""HTTP API: auth, validation, task lifecycle, webhook security."""
import hashlib
import hmac
import json
from unittest.mock import patch

import pytest

from app.core.config import get_settings

ISSUE = "https://github.com/psf/requests/issues/6655"


class TestHealth:
    def test_health_is_public_and_reports_config(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["retriever"] == "local"
        assert body["llm_configured"] is False

    def test_dashboard_and_login_pages_served(self, client):
        assert client.get("/").status_code == 200
        assert client.get("/login.html").status_code == 200


class TestAuth:
    def test_tasks_require_token(self, client):
        assert client.get("/api/v1/tasks").status_code == 401
        assert client.post("/api/v1/tasks", json={"github_issue_url": ISSUE}).status_code == 401

    def test_metrics_require_token(self, client):
        assert client.get("/api/v1/metrics").status_code == 401

    def test_garbage_token_rejected(self, client):
        assert client.get("/api/v1/tasks", headers={"Authorization": "Bearer nope"}).status_code == 401

    def test_me(self, client, auth):
        r = client.get("/api/v1/auth/me", headers=auth)
        assert r.status_code == 200 and r.json()["username"] == "engineer1"

    def test_wrong_password(self, client, auth):
        r = client.post("/api/v1/auth/login", json={"username": "engineer1", "password": "wrong-pass"})
        assert r.status_code == 401


class TestTasks:
    def test_create_queues_task(self, client, auth):
        with patch("app.api.tasks.run_pipeline") as run:
            r = client.post("/api/v1/tasks", json={"github_issue_url": ISSUE}, headers=auth)
        assert r.status_code == 202
        assert r.json()["status"] == "queued"
        run.assert_called_once()

    def test_rejects_non_issue_urls(self, client, auth):
        for bad in ["", "https://github.com/psf/requests", "https://evil.com/github.com/a/b/issues/1",
                    "https://github.com/psf/requests/pull/12"]:
            r = client.post("/api/v1/tasks", json={"github_issue_url": bad}, headers=auth)
            assert r.status_code == 422, bad

    def test_get_and_list(self, client, auth):
        with patch("app.api.tasks.run_pipeline"):
            tid = client.post("/api/v1/tasks", json={"github_issue_url": ISSUE}, headers=auth).json()["task_id"]
        assert client.get(f"/api/v1/tasks/{tid}", headers=auth).json()["task_id"] == tid
        listed = client.get("/api/v1/tasks?limit=1", headers=auth).json()
        assert isinstance(listed, list) and len(listed) == 1

    def test_unknown_and_malformed_ids_are_404(self, client, auth):
        assert client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000", headers=auth).status_code == 404
        assert client.get("/api/v1/tasks/not-a-uuid", headers=auth).status_code == 404

    def test_limit_is_bounded(self, client, auth):
        assert client.get("/api/v1/tasks?limit=100000", headers=auth).status_code == 422

    def test_worker_marks_task_failed_with_reason(self, client, auth):
        from app.api.tasks import run_pipeline
        with patch("app.api.tasks.run_pipeline"):
            tid = client.post("/api/v1/tasks", json={"github_issue_url": ISSUE}, headers=auth).json()["task_id"]
        with patch("app.api.tasks.fetch_github_issue", side_effect=RuntimeError("GitHub rate limit")):
            run_pipeline(tid)
        body = client.get(f"/api/v1/tasks/{tid}", headers=auth).json()
        assert body["status"] == "failed" and "GitHub rate limit" in body["error_message"]

    def test_metrics_shape(self, client, auth):
        r = client.get("/api/v1/metrics", headers=auth)
        assert r.status_code == 200 and "total_tasks" in r.json()["summary"]
        assert "data" in client.get("/api/v1/metrics/daily", headers=auth).json()


def _event(action="labeled", label="nexus"):
    return {"action": action, "label": {"name": label},
            "issue": {"number": 9, "title": "t", "html_url": ISSUE, "body": "b"},
            "repository": {"full_name": "psf/requests"}}


def _post_signed(client, payload, secret="shh", event="issues", sig=None):
    body = json.dumps(payload).encode()
    sig = sig or "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post("/api/v1/webhook/github", content=body, headers={
        "X-GitHub-Event": event, "Content-Type": "application/json", "X-Hub-Signature-256": sig})


class TestWebhook:
    def test_disabled_without_secret(self, client):
        r = client.post("/api/v1/webhook/github", json=_event(), headers={"X-GitHub-Event": "issues"})
        assert r.status_code == 503

    def test_bad_signature_rejected(self, client, monkeypatch):
        monkeypatch.setattr(get_settings(), "github_webhook_secret", "shh")
        assert _post_signed(client, _event(), sig="sha256=bad").status_code == 401

    def test_label_triggers_run(self, client, monkeypatch):
        monkeypatch.setattr(get_settings(), "github_webhook_secret", "shh")
        with patch("app.api.webhook.run_pipeline") as run:
            r = _post_signed(client, _event())
        assert r.json()["status"] == "triggered"
        run.assert_called_once()

    def test_opened_or_other_labels_do_not_spend_credits(self, client, monkeypatch):
        monkeypatch.setattr(get_settings(), "github_webhook_secret", "shh")
        with patch("app.api.webhook.run_pipeline") as run:
            assert _post_signed(client, _event(action="opened")).json()["status"] == "ignored"
            assert _post_signed(client, _event(label="bug")).json()["status"] == "ignored"
            assert _post_signed(client, {}, event="push").json()["status"] == "ignored"
        run.assert_not_called()


class TestOwnership:
    def test_users_only_see_their_own_tasks_admin_sees_all(self, client, auth, other, admin):
        with patch("app.api.tasks.run_pipeline"):
            mine = client.post("/api/v1/tasks", json={"github_issue_url": ISSUE}, headers=auth).json()["task_id"]
        assert client.get(f"/api/v1/tasks/{mine}", headers=other).status_code == 404
        assert mine not in [t["task_id"] for t in client.get("/api/v1/tasks?limit=100", headers=other).json()]
        assert client.get(f"/api/v1/tasks/{mine}", headers=admin).status_code == 200


class TestHardening:
    def test_login_throttled_after_five_failures(self, client):
        from tests.conftest import make_user
        make_user(client, "throttled")
        for _ in range(5):
            r = client.post("/api/v1/auth/login", json={"username": "throttled", "password": "nope-nope"})
            assert r.status_code == 401
        r = client.post("/api/v1/auth/login", json={"username": "throttled", "password": "s3cret-pass"})
        assert r.status_code == 429

    def test_production_refuses_default_secret_key(self, monkeypatch):
        import pydantic
        from app.core.config import Settings
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(pydantic.ValidationError, match="SECRET_KEY"):
            Settings(_env_file=None)
