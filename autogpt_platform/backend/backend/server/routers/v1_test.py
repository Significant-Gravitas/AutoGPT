import json
from unittest.mock import AsyncMock, Mock

import autogpt_libs.auth.depends
import fastapi
import fastapi.testclient
import pytest_mock
from pytest_snapshot.plugin import Snapshot

import backend.server.routers.v1 as v1_routes
from backend.data.credit import AutoTopUpConfig
from backend.data.graph import GraphModel
from backend.server.conftest import TEST_USER_ID
from backend.server.utils import get_user_id

app = fastapi.FastAPI()
app.include_router(v1_routes.v1_router)

client = fastapi.testclient.TestClient(app)


def override_auth_middleware(request: fastapi.Request) -> dict[str, str]:
    """Override auth middleware for testing"""
    return {"sub": TEST_USER_ID, "role": "user", "email": "test@example.com"}


def override_get_user_id() -> str:
    """Override get_user_id for testing"""
    return TEST_USER_ID


app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
    override_auth_middleware
)
app.dependency_overrides[get_user_id] = override_get_user_id


# Auth endpoints tests
def test_get_or_create_user_route(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test get or create user endpoint"""
    mock_user = Mock()
    mock_user.model_dump.return_value = {
        "id": TEST_USER_ID,
        "email": "test@example.com",
        "name": "Test User",
    }

    mocker.patch(
        "backend.server.routers.v1.get_or_create_user",
        return_value=mock_user,
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    response_data = response.json()

    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_user",
    )


def test_update_user_email_route(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test update user email endpoint"""
    mocker.patch(
        "backend.server.routers.v1.update_user_email",
        return_value=None,
    )

    response = client.post("/auth/user/email", json="newemail@example.com")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["email"] == "newemail@example.com"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_email",
    )


# Blocks endpoints tests
def test_get_graph_blocks(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get blocks endpoint"""
    # Mock block
    mock_block = Mock()
    mock_block.to_dict.return_value = {
        "id": "test-block",
        "name": "Test Block",
        "description": "A test block",
        "disabled": False,
    }
    mock_block.id = "test-block"
    mock_block.disabled = False

    # Mock get_blocks
    mocker.patch(
        "backend.server.routers.v1.get_blocks",
        return_value={"test-block": lambda: mock_block},
    )

    # Mock block costs
    mocker.patch(
        "backend.server.routers.v1.get_block_costs",
        return_value={"test-block": [{"cost": 10, "type": "credit"}]},
    )

    response = client.get("/blocks")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]["id"] == "test-block"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "blks_all",
    )


def test_execute_graph_block(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test execute block endpoint"""
    # Mock block
    mock_block = Mock()

    async def mock_execute(*args, **kwargs):
        yield "output1", {"data": "result1"}
        yield "output2", {"data": "result2"}

    mock_block.execute = mock_execute

    mocker.patch(
        "backend.server.routers.v1.get_block",
        return_value=mock_block,
    )

    request_data = {
        "input_name": "test_input",
        "input_value": "test_value",
    }

    response = client.post("/blocks/test-block/execute", json=request_data)

    assert response.status_code == 200
    response_data = response.json()

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "blks_exec",
    )


def test_execute_graph_block_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test execute block with non-existent block"""
    mocker.patch(
        "backend.server.routers.v1.get_block",
        return_value=None,
    )

    response = client.post("/blocks/nonexistent-block/execute", json={})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# Credits endpoints tests
def test_get_user_credits(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get user credits endpoint"""
    mock_credit_model = mocker.patch("backend.server.routers.v1._user_credit_model")
    mock_credit_model.get_credits = AsyncMock(return_value=1000)

    response = client.get("/credits")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["credits"] == 1000

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "cred_bal",
    )


def test_request_top_up(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test request top up endpoint"""
    mock_credit_model = mocker.patch("backend.server.routers.v1._user_credit_model")
    mock_credit_model.top_up_intent = AsyncMock(
        return_value="https://checkout.example.com/session123"
    )

    request_data = {"credit_amount": 500}

    response = client.post("/credits", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert "checkout_url" in response_data

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "cred_topup_req",
    )


def test_get_auto_top_up(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get auto top-up configuration endpoint"""
    mock_config = AutoTopUpConfig(threshold=100, amount=500)

    mocker.patch(
        "backend.server.routers.v1.get_auto_top_up",
        return_value=mock_config,
    )

    response = client.get("/credits/auto-top-up")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["threshold"] == 100
    assert response_data["amount"] == 500

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "cred_topup_cfg",
    )


# Graphs endpoints tests
def test_get_graphs(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get graphs endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id="test-user-id",
    )

    mocker.patch(
        "backend.server.routers.v1.graph_db.list_graphs",
        return_value=[mock_graph],
    )

    response = client.get("/graphs")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]["id"] == "graph-123"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grphs_all",
    )


def test_get_graph(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get single graph endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id="test-user-id",
    )

    mocker.patch(
        "backend.server.routers.v1.graph_db.get_graph",
        return_value=mock_graph,
    )

    response = client.get("/graphs/graph-123")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["id"] == "graph-123"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grph_single",
    )


def test_get_graph_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test get graph with non-existent ID"""
    mocker.patch(
        "backend.server.routers.v1.graph_db.get_graph",
        return_value=None,
    )

    response = client.get("/graphs/nonexistent-graph")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_graph(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test delete graph endpoint"""
    # Mock active graph for deactivation
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id="test-user-id",
    )

    mocker.patch(
        "backend.server.routers.v1.graph_db.get_graph",
        return_value=mock_graph,
    )
    mocker.patch(
        "backend.server.routers.v1.on_graph_deactivate",
        return_value=None,
    )
    mocker.patch(
        "backend.server.routers.v1.graph_db.delete_graph",
        return_value=3,  # Number of versions deleted
    )

    response = client.delete("/graphs/graph-123")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["version_counts"] == 3

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "grphs_del",
    )


# Invalid request tests
def test_invalid_json_request() -> None:
    """Test endpoint with invalid JSON"""
    response = client.post(
        "/auth/user/email",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_missing_required_field() -> None:
    """Test endpoint with missing required field"""
    response = client.post("/credits", json={})  # Missing credit_amount
    assert response.status_code == 422
