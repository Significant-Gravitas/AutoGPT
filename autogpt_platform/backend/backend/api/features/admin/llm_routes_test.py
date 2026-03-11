import json
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pytest_snapshot.plugin import Snapshot

import backend.api.features.admin.llm_routes as llm_routes
from backend.server.v2.llm import model as llm_model
from backend.util.models import Pagination

app = fastapi.FastAPI()
app.include_router(llm_routes.router, prefix="/admin/llm")

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_list_llm_providers_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful listing of LLM providers"""
    # Mock the database function
    mock_providers = [
        {
            "id": "provider-1",
            "name": "openai",
            "display_name": "OpenAI",
            "description": "OpenAI LLM provider",
            "supports_tools": True,
            "supports_json_output": True,
            "supports_reasoning": False,
            "supports_parallel_tool": True,
            "metadata": {},
            "models": [],
        },
        {
            "id": "provider-2",
            "name": "anthropic",
            "display_name": "Anthropic",
            "description": "Anthropic LLM provider",
            "supports_tools": True,
            "supports_json_output": True,
            "supports_reasoning": False,
            "supports_parallel_tool": True,
            "metadata": {},
            "models": [],
        },
    ]

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.list_providers",
        new=AsyncMock(return_value=mock_providers),
    )

    response = client.get("/admin/llm/providers")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["providers"]) == 2
    assert response_data["providers"][0]["name"] == "openai"

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "list_llm_providers_success.json",
    )


def test_list_llm_models_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful listing of LLM models with pagination"""
    # Mock the database function - now returns LlmModelsResponse
    mock_model = llm_model.LlmModel(
        id="model-1",
        slug="gpt-4o",
        display_name="GPT-4o",
        description="GPT-4 Optimized",
        provider_id="provider-1",
        context_window=128000,
        max_output_tokens=16384,
        is_enabled=True,
        capabilities={},
        metadata={},
        costs=[
            llm_model.LlmModelCost(
                id="cost-1",
                credit_cost=10,
                credential_provider="openai",
                metadata={},
            )
        ],
    )

    mock_response = llm_model.LlmModelsResponse(
        models=[mock_model],
        pagination=Pagination(
            total_items=1,
            total_pages=1,
            current_page=1,
            page_size=50,
        ),
    )

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.list_models",
        new=AsyncMock(return_value=mock_response),
    )

    response = client.get("/admin/llm/models")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["models"]) == 1
    assert response_data["models"][0]["slug"] == "gpt-4o"
    assert response_data["pagination"]["total_items"] == 1
    assert response_data["pagination"]["page_size"] == 50

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "list_llm_models_success.json",
    )


def test_create_llm_provider_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful creation of LLM provider"""
    mock_provider = {
        "id": "new-provider-id",
        "name": "groq",
        "display_name": "Groq",
        "description": "Groq LLM provider",
        "supports_tools": True,
        "supports_json_output": True,
        "supports_reasoning": False,
        "supports_parallel_tool": False,
        "metadata": {},
    }

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.upsert_provider",
        new=AsyncMock(return_value=mock_provider),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    request_data = {
        "name": "groq",
        "display_name": "Groq",
        "description": "Groq LLM provider",
        "supports_tools": True,
        "supports_json_output": True,
        "supports_reasoning": False,
        "supports_parallel_tool": False,
        "metadata": {},
    }

    response = client.post("/admin/llm/providers", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["name"] == "groq"
    assert response_data["display_name"] == "Groq"

    # Verify refresh was called
    mock_refresh.assert_called_once()

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "create_llm_provider_success.json",
    )


def test_create_llm_model_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful creation of LLM model"""
    mock_model = {
        "id": "new-model-id",
        "slug": "gpt-4.1-mini",
        "display_name": "GPT-4.1 Mini",
        "description": "Latest GPT-4.1 Mini model",
        "provider_id": "provider-1",
        "context_window": 128000,
        "max_output_tokens": 16384,
        "is_enabled": True,
        "capabilities": {},
        "metadata": {},
        "costs": [
            {
                "id": "cost-id",
                "credit_cost": 5,
                "credential_provider": "openai",
                "metadata": {},
            }
        ],
    }

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.create_model",
        new=AsyncMock(return_value=mock_model),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    request_data = {
        "slug": "gpt-4.1-mini",
        "display_name": "GPT-4.1 Mini",
        "description": "Latest GPT-4.1 Mini model",
        "provider_id": "provider-1",
        "context_window": 128000,
        "max_output_tokens": 16384,
        "is_enabled": True,
        "capabilities": {},
        "metadata": {},
        "costs": [
            {
                "credit_cost": 5,
                "credential_provider": "openai",
                "metadata": {},
            }
        ],
    }

    response = client.post("/admin/llm/models", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["slug"] == "gpt-4.1-mini"
    assert response_data["is_enabled"] is True

    # Verify refresh was called
    mock_refresh.assert_called_once()

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "create_llm_model_success.json",
    )


def test_update_llm_model_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful update of LLM model"""
    mock_model = {
        "id": "model-1",
        "slug": "gpt-4o",
        "display_name": "GPT-4o Updated",
        "description": "Updated description",
        "provider_id": "provider-1",
        "context_window": 256000,
        "max_output_tokens": 32768,
        "is_enabled": True,
        "capabilities": {},
        "metadata": {},
        "costs": [
            {
                "id": "cost-1",
                "credit_cost": 15,
                "credential_provider": "openai",
                "metadata": {},
            }
        ],
    }

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.update_model",
        new=AsyncMock(return_value=mock_model),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    request_data = {
        "display_name": "GPT-4o Updated",
        "description": "Updated description",
        "context_window": 256000,
        "max_output_tokens": 32768,
    }

    response = client.patch("/admin/llm/models/model-1", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["display_name"] == "GPT-4o Updated"
    assert response_data["context_window"] == 256000

    # Verify refresh was called
    mock_refresh.assert_called_once()

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "update_llm_model_success.json",
    )


def test_toggle_llm_model_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful toggling of LLM model enabled status"""
    # Create a proper mock model object
    mock_model = llm_model.LlmModel(
        id="model-1",
        slug="gpt-4o",
        display_name="GPT-4o",
        description="GPT-4 Optimized",
        provider_id="provider-1",
        context_window=128000,
        max_output_tokens=16384,
        is_enabled=False,
        capabilities={},
        metadata={},
        costs=[],
    )

    # Create a proper ToggleLlmModelResponse
    mock_response = llm_model.ToggleLlmModelResponse(
        model=mock_model,
        nodes_migrated=0,
        migrated_to_slug=None,
        migration_id=None,
    )

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.toggle_model",
        new=AsyncMock(return_value=mock_response),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    request_data = {"is_enabled": False}

    response = client.patch("/admin/llm/models/model-1/toggle", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["model"]["is_enabled"] is False

    # Verify refresh was called
    mock_refresh.assert_called_once()

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "toggle_llm_model_success.json",
    )


def test_delete_llm_model_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful deletion of LLM model with migration"""
    # Create a proper DeleteLlmModelResponse
    mock_response = llm_model.DeleteLlmModelResponse(
        deleted_model_slug="gpt-3.5-turbo",
        deleted_model_display_name="GPT-3.5 Turbo",
        replacement_model_slug="gpt-4o-mini",
        nodes_migrated=42,
        message="Successfully deleted model 'GPT-3.5 Turbo' (gpt-3.5-turbo) "
        "and migrated 42 workflow node(s) to 'gpt-4o-mini'.",
    )

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.delete_model",
        new=AsyncMock(return_value=mock_response),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    response = client.delete(
        "/admin/llm/models/model-1?replacement_model_slug=gpt-4o-mini"
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["deleted_model_slug"] == "gpt-3.5-turbo"
    assert response_data["nodes_migrated"] == 42
    assert response_data["replacement_model_slug"] == "gpt-4o-mini"

    # Verify refresh was called
    mock_refresh.assert_called_once()

    # Snapshot test the response (must be string)
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "delete_llm_model_success.json",
    )


def test_delete_llm_model_validation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test deletion fails with proper error when validation fails"""
    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.delete_model",
        new=AsyncMock(side_effect=ValueError("Replacement model 'invalid' not found")),
    )

    response = client.delete("/admin/llm/models/model-1?replacement_model_slug=invalid")

    assert response.status_code == 400
    assert "Replacement model 'invalid' not found" in response.json()["detail"]


def test_delete_llm_model_no_replacement_with_usage(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test deletion fails when nodes exist but no replacement is provided"""
    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.delete_model",
        new=AsyncMock(
            side_effect=ValueError(
                "Cannot delete model 'test-model': 5 workflow node(s) are using it. "
                "Please provide a replacement_model_slug to migrate them."
            )
        ),
    )

    response = client.delete("/admin/llm/models/model-1")

    assert response.status_code == 400
    assert "workflow node(s) are using it" in response.json()["detail"]


def test_delete_llm_model_no_replacement_no_usage(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test deletion succeeds when no nodes use the model and no replacement is provided"""
    mock_response = llm_model.DeleteLlmModelResponse(
        deleted_model_slug="unused-model",
        deleted_model_display_name="Unused Model",
        replacement_model_slug=None,
        nodes_migrated=0,
        message="Successfully deleted model 'Unused Model' (unused-model). No workflows were using this model.",
    )

    mocker.patch(
        "backend.api.features.admin.llm_routes.llm_db.delete_model",
        new=AsyncMock(return_value=mock_response),
    )

    mock_refresh = mocker.patch(
        "backend.api.features.admin.llm_routes._refresh_runtime_state",
        new=AsyncMock(),
    )

    response = client.delete("/admin/llm/models/model-1")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["deleted_model_slug"] == "unused-model"
    assert response_data["nodes_migrated"] == 0
    assert response_data["replacement_model_slug"] is None
    mock_refresh.assert_called_once()
