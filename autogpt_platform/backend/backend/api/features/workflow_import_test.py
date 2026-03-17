"""Tests for workflow_import.py API endpoint."""

from unittest.mock import AsyncMock

import fastapi
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient

from backend.api.features.workflow_import import router

app = fastapi.FastAPI()
app.include_router(router)
client = TestClient(app)

# Sample workflow fixtures
N8N_WORKFLOW = {
    "name": "Email on Webhook",
    "nodes": [
        {
            "name": "Webhook",
            "type": "n8n-nodes-base.webhookTrigger",
            "parameters": {"path": "/incoming"},
        },
        {
            "name": "Send Email",
            "type": "n8n-nodes-base.gmail",
            "parameters": {"resource": "message", "operation": "send"},
        },
    ],
    "connections": {
        "Webhook": {"main": [[{"node": "Send Email", "type": "main", "index": 0}]]}
    },
}

MAKE_WORKFLOW = {
    "name": "Sheets to Calendar",
    "flow": [
        {
            "module": "google-sheets:watchUpdatedCells",
            "mapper": {"spreadsheetId": "abc"},
        },
        {
            "module": "google-calendar:createAnEvent",
            "mapper": {"title": "Meeting"},
        },
    ],
}

ZAPIER_WORKFLOW = {
    "name": "Gmail to Slack",
    "steps": [
        {"app": "Gmail", "action": "new_email"},
        {"app": "Slack", "action": "send_message", "params": {"channel": "#alerts"}},
    ],
}


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


class TestImportWorkflow:
    def test_import_n8n_workflow(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "n8n"
        assert data["source_name"] == "Email on Webhook"
        assert "copilot_prompt" in data
        assert "n8n" in data["copilot_prompt"]
        assert "Email on Webhook" in data["copilot_prompt"]

    def test_import_make_workflow(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": MAKE_WORKFLOW},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "make"
        assert data["source_name"] == "Sheets to Calendar"
        assert "copilot_prompt" in data

    def test_import_zapier_workflow(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": ZAPIER_WORKFLOW},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "zapier"
        assert data["source_name"] == "Gmail to Slack"
        assert "copilot_prompt" in data

    def test_prompt_includes_steps(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW},
        )
        prompt = response.json()["copilot_prompt"]
        # Should include step details from the workflow
        assert "Webhook" in prompt or "webhook" in prompt
        assert "Gmail" in prompt or "gmail" in prompt

    def test_no_source_provided(self):
        response = client.post(
            "/workflow",
            json={},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_both_sources_provided(self):
        response = client.post(
            "/workflow",
            json={
                "workflow_json": N8N_WORKFLOW,
                "template_url": "https://n8n.io/workflows/123",
            },
        )
        assert response.status_code == 422

    def test_unknown_format_returns_400(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": {"foo": "bar"}},
        )
        assert response.status_code == 400
        assert "Could not detect workflow format" in response.json()["detail"]

    def test_url_fetch_bad_url_returns_400(self, mocker):
        mocker.patch(
            "backend.api.features.workflow_import.fetch_n8n_template",
            new_callable=AsyncMock,
            side_effect=ValueError("Invalid URL format"),
        )
        response = client.post(
            "/workflow",
            json={"template_url": "https://bad-url.com"},
        )
        assert response.status_code == 400
        assert "Invalid URL format" in response.json()["detail"]

    def test_url_fetch_upstream_error_returns_502(self, mocker):
        mocker.patch(
            "backend.api.features.workflow_import.fetch_n8n_template",
            new_callable=AsyncMock,
            side_effect=RuntimeError("n8n API returned 500"),
        )
        response = client.post(
            "/workflow",
            json={"template_url": "https://n8n.io/workflows/123"},
        )
        assert response.status_code == 502
        assert "n8n API returned 500" in response.json()["detail"]

    def test_response_model_shape(self):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW},
        )
        data = response.json()
        assert "copilot_prompt" in data
        assert "source_format" in data
        assert "source_name" in data
        assert isinstance(data["copilot_prompt"], str)
        assert len(data["copilot_prompt"]) > 0
