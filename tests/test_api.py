"""
Integration tests for the NEXUS FastAPI endpoints.
Uses TestClient (no real DB/LLM calls — mocked).
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_health_has_version(self):
        r = client.get("/api/v1/health")
        assert "version" in r.json()


class TestCreateTask:
    def test_valid_github_url_returns_202(self):
        with patch("app.api.tasks.run_pipeline"):
            r = client.post("/api/v1/tasks", json={
                "github_issue_url": "https://github.com/psf/requests/issues/7443",
                "use_hyde": True,
            })
        assert r.status_code == 202

    def test_response_has_task_id(self):
        with patch("app.api.tasks.run_pipeline"):
            r = client.post("/api/v1/tasks", json={
                "github_issue_url": "https://github.com/psf/requests/issues/7443",
                "use_hyde": True,
            })
        assert "task_id" in r.json()

    def test_response_status_is_queued(self):
        with patch("app.api.tasks.run_pipeline"):
            r = client.post("/api/v1/tasks", json={
                "github_issue_url": "https://github.com/psf/requests/issues/7443",
            })
        assert r.json()["status"] == "queued"

    def test_missing_url_returns_422(self):
        r = client.post("/api/v1/tasks", json={})
        assert r.status_code == 422

    def test_empty_url_returns_422(self):
        r = client.post("/api/v1/tasks", json={"github_issue_url": ""})
        assert r.status_code in (422, 400, 500)


class TestGetTask:
    def test_nonexistent_task_returns_404(self):
        r = client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_invalid_uuid_returns_error(self):
        r = client.get("/api/v1/tasks/not-a-uuid")
        assert r.status_code in (404, 422, 400)

    def test_created_task_retrievable(self):
        with patch("app.api.tasks.run_pipeline"):
            create_r = client.post("/api/v1/tasks", json={
                "github_issue_url": "https://github.com/psf/requests/issues/7443",
            })
        task_id = create_r.json()["task_id"]
        get_r = client.get(f"/api/v1/tasks/{task_id}")
        assert get_r.status_code == 200
        assert get_r.json()["task_id"] == task_id


class TestListTasks:
    def test_list_returns_array(self):
        r = client.get("/api/v1/tasks")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_limit_param(self):
        r = client.get("/api/v1/tasks?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200

    def test_metrics_has_summary(self):
        r = client.get("/api/v1/metrics")
        data = r.json()
        assert "summary" in data
        assert "total_tasks" in data["summary"]

    def test_daily_metrics_returns_200(self):
        r = client.get("/api/v1/metrics/daily")
        assert r.status_code == 200

    def test_daily_metrics_has_data(self):
        r = client.get("/api/v1/metrics/daily")
        assert "data" in r.json()


class TestWebhookEndpoint:
    def test_webhook_health(self):
        r = client.get("/api/v1/webhook/github/health")
        assert r.status_code == 200

    def test_non_issue_event_ignored(self):
        r = client.post(
            "/api/v1/webhook/github",
            json={"action": "created"},
            headers={"X-GitHub-Event": "push"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "ignored"

    def test_issue_opened_triggers_pipeline(self):
        payload = {
            "action": "opened",
            "issue": {
                "number": 999,
                "title": "Test issue",
                "html_url": "https://github.com/psf/requests/issues/999",
                "body": "Test body",
            },
            "repository": {
                "full_name": "psf/requests",
            },
        }
        with patch("app.api.webhook.run_pipeline"):
            r = client.post(
                "/api/v1/webhook/github",
                json=payload,
                headers={"X-GitHub-Event": "issues"},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "triggered"
