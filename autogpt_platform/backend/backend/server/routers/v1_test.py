import json
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
import starlette.datastructures
from fastapi import HTTPException, UploadFile
from pytest_snapshot.plugin import Snapshot

import backend.server.routers.v1 as v1_routes
from backend.data.credit import AutoTopUpConfig
from backend.data.graph import GraphModel
from backend.server.routers.v1 import upload_file

app = fastapi.FastAPI()
app.include_router(v1_routes.v1_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, setup_test_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    # setup_test_user fixture already executed and user is created in database
    # It returns the user_id which we don't need to await

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# Auth endpoints tests
def test_get_or_create_user_route(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get or create user endpoint"""
    mock_user = Mock()
    mock_user.model_dump.return_value = {
        "id": test_user_id,
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
        "backend.data.credit.get_block_cost",
        return_value=[{"cost": 10, "type": "credit"}],
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

    # Mock user for user_context
    mock_user = Mock()
    mock_user.timezone = "UTC"

    mocker.patch(
        "backend.server.routers.v1.get_user_by_id",
        return_value=mock_user,
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
    mock_credit_model = Mock()
    mock_credit_model.get_credits = AsyncMock(return_value=1000)
    mocker.patch(
        "backend.server.routers.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

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
    mock_credit_model = Mock()
    mock_credit_model.top_up_intent = AsyncMock(
        return_value="https://checkout.example.com/session123"
    )
    mocker.patch(
        "backend.server.routers.v1.get_user_credit_model",
        return_value=mock_credit_model,
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


def test_configure_auto_top_up(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test configure auto top-up endpoint - this test would have caught the enum casting bug"""
    # Mock the set_auto_top_up function to avoid database operations
    mocker.patch(
        "backend.server.routers.v1.set_auto_top_up",
        return_value=None,
    )

    # Mock credit model to avoid Stripe API calls
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50  # Current balance below threshold
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.server.routers.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    # Test data
    request_data = {
        "threshold": 100,
        "amount": 500,
    }

    response = client.post("/credits/auto-top-up", json=request_data)

    # This should succeed with our fix, but would have failed before with the enum casting error
    assert response.status_code == 200
    assert response.json() == "Auto top-up settings updated"


def test_configure_auto_top_up_validation_errors(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test configure auto top-up endpoint validation"""
    # Mock set_auto_top_up to avoid database operations for successful case
    mocker.patch("backend.server.routers.v1.set_auto_top_up")

    # Mock credit model to avoid Stripe API calls for the successful case
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.server.routers.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    # Test negative threshold
    response = client.post(
        "/credits/auto-top-up", json={"threshold": -1, "amount": 500}
    )
    assert response.status_code == 422  # Validation error

    # Test amount too small (but not 0)
    response = client.post(
        "/credits/auto-top-up", json={"threshold": 100, "amount": 100}
    )
    assert response.status_code == 422  # Validation error

    # Test amount = 0 (should be allowed)
    response = client.post("/credits/auto-top-up", json={"threshold": 100, "amount": 0})
    assert response.status_code == 200  # Should succeed


# Graphs endpoints tests
def test_get_graphs(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get graphs endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )

    mocker.patch(
        "backend.data.graph.list_graphs_paginated",
        return_value=Mock(graphs=[mock_graph]),
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
    test_user_id: str,
) -> None:
    """Test get single graph endpoint"""
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
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
    test_user_id: str,
) -> None:
    """Test delete graph endpoint"""
    # Mock active graph for deactivation
    mock_graph = GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=test_user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
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


@pytest.mark.asyncio
async def test_upload_file_success(test_user_id: str):
    """Test successful file upload."""
    # Create mock upload file
    file_content = b"test file content"
    file_obj = BytesIO(file_content)
    upload_file_mock = UploadFile(
        filename="test.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    # Mock dependencies
    with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
        "backend.server.routers.v1.get_cloud_storage_handler"
    ) as mock_handler_getter:

        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.return_value = "gcs://test-bucket/uploads/123/test.txt"
        mock_handler_getter.return_value = mock_handler

        # Mock file.read()
        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(
            file=upload_file_mock,
            user_id=test_user_id,
            provider="gcs",
            expiration_hours=24,
        )

        # Verify result
        assert result.file_uri == "gcs://test-bucket/uploads/123/test.txt"
        assert result.file_name == "test.txt"
        assert result.size == len(file_content)
        assert result.content_type == "text/plain"
        assert result.expires_in_hours == 24

        # Verify virus scan was called
        mock_scan.assert_called_once_with(file_content, filename="test.txt")

        # Verify cloud storage operations
        mock_handler.store_file.assert_called_once_with(
            content=file_content,
            filename="test.txt",
            provider="gcs",
            expiration_hours=24,
            user_id=test_user_id,
        )


@pytest.mark.asyncio
async def test_upload_file_no_filename(test_user_id: str):
    """Test file upload without filename."""
    file_content = b"test content"
    file_obj = BytesIO(file_content)
    upload_file_mock = UploadFile(
        filename=None,
        file=file_obj,
        headers=starlette.datastructures.Headers(
            {"content-type": "application/octet-stream"}
        ),
    )

    with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
        "backend.server.routers.v1.get_cloud_storage_handler"
    ) as mock_handler_getter:

        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.return_value = (
            "gcs://test-bucket/uploads/123/uploaded_file"
        )
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(file=upload_file_mock, user_id=test_user_id)

        assert result.file_name == "uploaded_file"
        assert result.content_type == "application/octet-stream"

        # Verify virus scan was called with default filename
        mock_scan.assert_called_once_with(file_content, filename="uploaded_file")


@pytest.mark.asyncio
async def test_upload_file_invalid_expiration(test_user_id: str):
    """Test file upload with invalid expiration hours."""
    file_obj = BytesIO(b"content")
    upload_file_mock = UploadFile(
        filename="test.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    # Test expiration too short
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(
            file=upload_file_mock, user_id=test_user_id, expiration_hours=0
        )
    assert exc_info.value.status_code == 400
    assert "between 1 and 48" in exc_info.value.detail

    # Test expiration too long
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(
            file=upload_file_mock, user_id=test_user_id, expiration_hours=49
        )
    assert exc_info.value.status_code == 400
    assert "between 1 and 48" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_file_virus_scan_failure(test_user_id: str):
    """Test file upload when virus scan fails."""
    file_content = b"malicious content"
    file_obj = BytesIO(file_content)
    upload_file_mock = UploadFile(
        filename="virus.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with patch("backend.server.routers.v1.scan_content_safe") as mock_scan:
        # Mock virus scan to raise exception
        mock_scan.side_effect = RuntimeError("Virus detected!")

        upload_file_mock.read = AsyncMock(return_value=file_content)

        with pytest.raises(RuntimeError, match="Virus detected!"):
            await upload_file(file=upload_file_mock, user_id=test_user_id)


@pytest.mark.asyncio
async def test_upload_file_cloud_storage_failure(test_user_id: str):
    """Test file upload when cloud storage fails."""
    file_content = b"test content"
    file_obj = BytesIO(file_content)
    upload_file_mock = UploadFile(
        filename="test.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
        "backend.server.routers.v1.get_cloud_storage_handler"
    ) as mock_handler_getter:

        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.side_effect = RuntimeError("Storage error!")
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        with pytest.raises(RuntimeError, match="Storage error!"):
            await upload_file(file=upload_file_mock, user_id=test_user_id)


@pytest.mark.asyncio
async def test_upload_file_size_limit_exceeded(test_user_id: str):
    """Test file upload when file size exceeds the limit."""
    # Create a file that exceeds the default 256MB limit
    large_file_content = b"x" * (257 * 1024 * 1024)  # 257MB
    file_obj = BytesIO(large_file_content)
    upload_file_mock = UploadFile(
        filename="large_file.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    upload_file_mock.read = AsyncMock(return_value=large_file_content)

    with pytest.raises(HTTPException) as exc_info:
        await upload_file(file=upload_file_mock, user_id=test_user_id)

    assert exc_info.value.status_code == 400
    assert "exceeds the maximum allowed size of 256MB" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_file_gcs_not_configured_fallback(test_user_id: str):
    """Test file upload fallback to base64 when GCS is not configured."""
    file_content = b"test file content"
    file_obj = BytesIO(file_content)
    upload_file_mock = UploadFile(
        filename="test.txt",
        file=file_obj,
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with patch("backend.server.routers.v1.scan_content_safe") as mock_scan, patch(
        "backend.server.routers.v1.get_cloud_storage_handler"
    ) as mock_handler_getter:

        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.config.gcs_bucket_name = ""  # Simulate no GCS bucket configured
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(file=upload_file_mock, user_id=test_user_id)

        # Verify fallback behavior
        assert result.file_name == "test.txt"
        assert result.size == len(file_content)
        assert result.content_type == "text/plain"
        assert result.expires_in_hours == 24

        # Verify file_uri is base64 data URI
        expected_data_uri = "data:text/plain;base64,dGVzdCBmaWxlIGNvbnRlbnQ="
        assert result.file_uri == expected_data_uri

        # Verify virus scan was called
        mock_scan.assert_called_once_with(file_content, filename="test.txt")

        # Verify cloud storage methods were NOT called
        mock_handler.store_file.assert_not_called()
