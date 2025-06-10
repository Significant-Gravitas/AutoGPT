import json

import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import fastapi
import fastapi.testclient
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.v2.otto.models as otto_models
import backend.server.v2.otto.routes as otto_routes
from backend.server.utils import get_user_id
from backend.server.v2.otto.service import OttoService

app = fastapi.FastAPI()
app.include_router(otto_routes.router)

client = fastapi.testclient.TestClient(app)


def override_auth_middleware():
    """Override auth middleware for testing"""
    return {"sub": "test-user-id"}


def override_get_user_id():
    """Override get_user_id for testing"""
    return "test-user-id"


app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
    override_auth_middleware
)
app.dependency_overrides[get_user_id] = override_get_user_id


def test_ask_otto_success(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test successful Otto API request"""
    # Mock the OttoService.ask method
    mock_response = otto_models.ApiResponse(
        answer="This is Otto's response to your query.",
        documents=[
            otto_models.Document(
                url="https://example.com/doc1",
                relevance_score=0.95,
            ),
            otto_models.Document(
                url="https://example.com/doc2",
                relevance_score=0.87,
            ),
        ],
        success=True,
    )

    mocker.patch.object(
        OttoService,
        "ask",
        return_value=mock_response,
    )

    request_data = {
        "query": "How do I create an agent?",
        "conversation_history": [
            {
                "query": "What is AutoGPT?",
                "response": "AutoGPT is an AI agent platform.",
            }
        ],
        "message_id": "msg_123",
        "include_graph_data": False,
    }

    response = client.post("/ask", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["answer"] == "This is Otto's response to your query."
    assert len(response_data["documents"]) == 2

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "otto_ok",
    )


def test_ask_otto_with_graph_data(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test Otto API request with graph data included"""
    # Mock the OttoService.ask method
    mock_response = otto_models.ApiResponse(
        answer="Here's information about your graph.",
        documents=[
            otto_models.Document(
                url="https://example.com/graph-doc",
                relevance_score=0.92,
            ),
        ],
        success=True,
    )

    mocker.patch.object(
        OttoService,
        "ask",
        return_value=mock_response,
    )

    request_data = {
        "query": "Tell me about my graph",
        "conversation_history": [],
        "message_id": "msg_456",
        "include_graph_data": True,
        "graph_id": "graph_123",
    }

    response = client.post("/ask", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "otto_grph",
    )


def test_ask_otto_empty_conversation(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test Otto API request with empty conversation history"""
    # Mock the OttoService.ask method
    mock_response = otto_models.ApiResponse(
        answer="Welcome! How can I help you?",
        documents=[],
        success=True,
    )

    mocker.patch.object(
        OttoService,
        "ask",
        return_value=mock_response,
    )

    request_data = {
        "query": "Hello",
        "conversation_history": [],
        "message_id": "msg_789",
        "include_graph_data": False,
    }

    response = client.post("/ask", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert len(response_data["documents"]) == 0

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "otto_empty",
    )


def test_ask_otto_service_error(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test Otto API request when service returns error"""
    # Mock the OttoService.ask method to return failure
    mock_response = otto_models.ApiResponse(
        answer="An error occurred while processing your request.",
        documents=[],
        success=False,
    )

    mocker.patch.object(
        OttoService,
        "ask",
        return_value=mock_response,
    )

    request_data = {
        "query": "Test query",
        "conversation_history": [],
        "message_id": "msg_error",
        "include_graph_data": False,
    }

    response = client.post("/ask", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is False

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "otto_err",
    )


def test_ask_otto_invalid_request() -> None:
    """Test Otto API with invalid request data"""
    # Missing required fields
    response = client.post("/ask", json={})
    assert response.status_code == 422

    # Invalid conversation history format
    response = client.post(
        "/ask",
        json={
            "query": "Test",
            "conversation_history": "not a list",
            "message_id": "123",
        },
    )
    assert response.status_code == 422

    # Missing message_id
    response = client.post(
        "/ask",
        json={
            "query": "Test",
            "conversation_history": [],
        },
    )
    assert response.status_code == 422


def test_ask_otto_unauthenticated(mocker: pytest_mock.MockFixture) -> None:
    """Test Otto API request without authentication"""
    # Remove the auth override to test unauthenticated access
    app.dependency_overrides.clear()

    # Mock auth_middleware to raise an exception
    mocker.patch(
        "autogpt_libs.auth.middleware.auth_middleware",
        side_effect=fastapi.HTTPException(status_code=401, detail="Unauthorized"),
    )

    request_data = {
        "query": "Test",
        "conversation_history": [],
        "message_id": "123",
    }

    response = client.post("/ask", json=request_data)
    # When auth is disabled and Otto API URL is not configured, we get 503
    assert response.status_code == 503

    # Restore the override
    app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
        override_auth_middleware
    )
    app.dependency_overrides[autogpt_libs.auth.depends.get_user_id] = (
        override_get_user_id
    )
