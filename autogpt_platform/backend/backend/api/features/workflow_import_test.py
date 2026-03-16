"""Tests for workflow_import.py API endpoint."""

from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture()
def mock_converter(mocker):
    """Mock the LLM converter to avoid actual LLM calls."""
    agent_json = {
        "name": "Converted Agent",
        "description": "Test agent",
        "version": 1,
        "is_active": True,
        "nodes": [],
        "links": [],
    }
    return mocker.patch(
        "backend.api.features.workflow_import.convert_workflow",
        new_callable=AsyncMock,
        return_value=(agent_json, ["Applied 2 auto-fixes"]),
    )


@pytest.fixture()
def mock_save(mocker):
    """Mock save_agent_to_library."""
    graph = MagicMock()
    graph.id = "graph-123"
    graph.name = "Converted Agent"
    library_agent = MagicMock()
    library_agent.id = "lib-456"
    return mocker.patch(
        "backend.copilot.tools.agent_generator.core.save_agent_to_library",
        new_callable=AsyncMock,
        return_value=(graph, library_agent),
    )


class TestImportWorkflow:
    def test_import_n8n_workflow(self, mock_converter, mock_save):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW, "save": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "n8n"
        assert data["source_name"] == "Email on Webhook"
        assert data["graph_id"] == "graph-123"
        mock_converter.assert_called_once()

    def test_import_make_workflow(self, mock_converter, mock_save):
        response = client.post(
            "/workflow",
            json={"workflow_json": MAKE_WORKFLOW, "save": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "make"
        assert data["source_name"] == "Sheets to Calendar"

    def test_import_zapier_workflow(self, mock_converter, mock_save):
        response = client.post(
            "/workflow",
            json={"workflow_json": ZAPIER_WORKFLOW, "save": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_format"] == "zapier"
        assert data["source_name"] == "Gmail to Slack"

    def test_import_without_save(self, mock_converter, mock_save):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW, "save": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["graph_id"] is None
        assert data["library_agent_id"] is None
        mock_save.assert_not_called()

    def test_no_source_provided(self):
        response = client.post(
            "/workflow",
            json={"save": True},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_both_sources_provided(self):
        response = client.post(
            "/workflow",
            json={
                "workflow_json": N8N_WORKFLOW,
                "template_url": "https://n8n.io/workflows/123",
                "save": True,
            },
        )
        assert response.status_code == 422

    def test_unknown_format_returns_400(self, mock_converter):
        response = client.post(
            "/workflow",
            json={"workflow_json": {"foo": "bar"}, "save": False},
        )
        assert response.status_code == 400
        assert "Could not detect workflow format" in response.json()["detail"]

    def test_converter_failure_returns_502(self, mocker):
        mocker.patch(
            "backend.api.features.workflow_import.convert_workflow",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        )
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW, "save": False},
        )
        assert response.status_code == 502
        assert "LLM call failed" in response.json()["detail"]

    def test_save_failure_returns_500(self, mock_converter, mocker):
        mocker.patch(
            "backend.copilot.tools.agent_generator.core.save_agent_to_library",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB connection failed"),
        )
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW, "save": True},
        )
        assert response.status_code == 500
        assert "could not be saved" in response.json()["detail"]

    def test_url_fetch_bad_url_returns_400(self, mocker):
        mocker.patch(
            "backend.api.features.workflow_import.fetch_n8n_template",
            new_callable=AsyncMock,
            side_effect=ValueError("Invalid URL format"),
        )
        response = client.post(
            "/workflow",
            json={"template_url": "https://bad-url.com", "save": False},
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
            json={"template_url": "https://n8n.io/workflows/123", "save": False},
        )
        assert response.status_code == 502
        assert "n8n API returned 500" in response.json()["detail"]

    def test_response_model_shape(self, mock_converter, mock_save):
        response = client.post(
            "/workflow",
            json={"workflow_json": N8N_WORKFLOW, "save": True},
        )
        data = response.json()
        # Verify all expected fields are present
        assert "graph" in data
        assert "graph_id" in data
        assert "library_agent_id" in data
        assert "source_format" in data
        assert "source_name" in data
        assert "conversion_notes" in data
        assert isinstance(data["conversion_notes"], list)
