import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
import starlette.datastructures
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException, UploadFile
from pytest_snapshot.plugin import Snapshot

from backend.api.rest_api import handle_internal_http_error
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import InsufficientBalanceError

from .v1 import upload_file, v1_router


def _test_ctx(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id="test-org",
        team_id="test-workspace",
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


app = fastapi.FastAPI()
app.include_router(v1_router)
# Mirror rest_api.py's GraphActivationError → 400 mapping so the atomicity
# tests below verify the same behaviour the real app exposes.
app.add_exception_handler(GraphActivationError, handle_internal_http_error(400))

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, setup_test_user, test_user_id):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    # setup_test_user fixture already executed and user is created in database
    # It returns the user_id which we don't need to await

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # Override get_request_context too — the real one queries Prisma to
    # resolve the user's personal org when no X-Org-Id header is set,
    # which closes/leaks the test event loop across sync TestClient calls.
    async def _fake_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _fake_request_context
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
    mock_user.created_at = datetime.now(timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
        "name": "Test User",
    }
    mock_result = Mock(user=mock_user, was_created=False)

    mocker.patch(
        "backend.api.features.v1.get_or_create_user_with_status",
        return_value=mock_result,
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "false"
    response_data = response.json()

    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_user",
    )


def test_get_or_create_user_route_reports_creation(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mock_user = Mock()
    mock_user.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
    }

    mocker.patch(
        "backend.api.features.v1.get_or_create_user_with_status",
        return_value=Mock(user=mock_user, was_created=True),
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "true"


def test_get_or_create_user_route_documents_creation_header() -> None:
    response_schema = app.openapi()["paths"]["/auth/user"]["post"]["responses"]["200"]

    assert response_schema["headers"]["X-AutoGPT-User-Created"] == {
        "description": "Whether this request created a new user",
        "schema": {"type": "string", "enum": ["true", "false"]},
    }


def test_update_user_email_route(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test update user email endpoint"""
    mocker.patch(
        "backend.api.features.v1.update_user_email",
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
        "backend.api.features.v1.get_blocks",
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
    mock_block.disabled = False
    mock_block.name = "TestBlock"

    async def mock_execute(*args, **kwargs):
        yield "output1", {"data": "result1"}
        yield "output2", {"data": "result2"}

    mock_block.execute = mock_execute

    mocker.patch(
        "backend.api.features.v1.get_block",
        return_value=mock_block,
    )

    # Mock user for user_context
    mock_user = Mock()
    mock_user.timezone = "UTC"

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        return_value=mock_user,
    )

    # Default to free block: cost = 0, no charge call.
    cost_mock = mocker.patch(
        "backend.api.features.v1.execution_utils.block_usage_cost",
        return_value=(0, {}),
    )
    mock_credit_model = mocker.AsyncMock()
    mocker.patch(
        "backend.api.features.v1.get_credit_model",
        return_value=mock_credit_model,
    )

    request_data = {
        "input_name": "test_input",
        "input_value": "test_value",
    }

    response = client.post("/blocks/test-block/execute", json=request_data)

    assert response.status_code == 200
    response_data = response.json()

    # Cost = 0 path: no spend_credits call.
    cost_mock.assert_called_once()
    mock_credit_model.spend_credits.assert_not_awaited()

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "blks_exec",
    )


def test_execute_graph_block_forwards_execution_context(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Regression for #12648: blocks that read execution_context (e.g. time
    blocks) crashed because the direct-block-execute route didn't forward
    one. The route must construct an ExecutionContext carrying the caller's
    user_id + timezone and pass it through to ``Block.execute``."""
    captured_kwargs: dict = {}

    mock_block = Mock()
    mock_block.disabled = False
    mock_block.name = "TestBlock"

    async def mock_execute(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield "output", {"data": "ok"}

    mock_block.execute = mock_execute

    mocker.patch(
        "backend.api.features.v1.get_block",
        return_value=mock_block,
    )

    mock_user = Mock()
    mock_user.timezone = "America/New_York"
    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        return_value=mock_user,
    )

    mocker.patch(
        "backend.api.features.v1.execution_utils.block_usage_cost",
        return_value=(0, {}),
    )

    response = client.post("/blocks/test-block/execute", json={"x": "y"})

    assert response.status_code == 200
    assert "execution_context" in captured_kwargs
    ctx = captured_kwargs["execution_context"]
    assert ctx.user_id == test_user_id
    assert ctx.user_timezone == "America/New_York"


def test_execute_graph_block_charges_when_cost_positive(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Paid blocks must charge credits before executing."""
    mock_block = Mock()
    mock_block.disabled = False
    mock_block.id = "paid-block"
    mock_block.name = "PaidBlock"

    async def mock_execute(*args, **kwargs):
        yield "output", {"data": "ok"}

    mock_block.execute = mock_execute

    mocker.patch(
        "backend.api.features.v1.get_block",
        return_value=mock_block,
    )
    mock_user = Mock()
    mock_user.timezone = "UTC"
    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        return_value=mock_user,
    )

    cost_filter = {"model": "gpt-4"}
    mocker.patch(
        "backend.executor.utils.block_usage_cost",
        return_value=(42, cost_filter),
    )
    mock_credit_model = mocker.AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.post(
        "/blocks/paid-block/execute", json={"input_name": "x", "input_value": "y"}
    )

    assert response.status_code == 200
    mock_credit_model.spend_credits.assert_awaited_once()
    call_kwargs = mock_credit_model.spend_credits.await_args.kwargs
    assert call_kwargs["cost"] == 42
    metadata = call_kwargs["metadata"]
    assert metadata.block_id == "paid-block"
    assert metadata.block == "PaidBlock"
    assert metadata.input == cost_filter
    assert metadata.reason == "Direct internal block execution of PaidBlock"


def test_execute_graph_block_returns_402_on_insufficient_balance(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """If spend_credits raises InsufficientBalanceError, endpoint returns 402."""
    mock_block = Mock()
    mock_block.disabled = False
    mock_block.id = "paid-block"
    mock_block.name = "PaidBlock"
    mock_block.execute = AsyncMock()

    mocker.patch(
        "backend.api.features.v1.get_block",
        return_value=mock_block,
    )
    mock_user = Mock()
    mock_user.timezone = "UTC"
    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        return_value=mock_user,
    )
    mocker.patch(
        "backend.executor.utils.block_usage_cost",
        return_value=(99, {}),
    )

    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.spend_credits.side_effect = InsufficientBalanceError(
        message="Insufficient balance",
        user_id=test_user_id,
        balance=10,
        amount=99,
    )
    mocker.patch(
        "backend.executor.utils.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.post(
        "/blocks/paid-block/execute", json={"input_name": "x", "input_value": "y"}
    )

    assert response.status_code == 402
    mock_block.execute.assert_not_called()


def test_execute_graph_block_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test execute block with non-existent block"""
    mocker.patch(
        "backend.api.features.v1.get_block",
        return_value=None,
    )

    response = client.post("/blocks/nonexistent-block/execute", json={})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# Invalid request tests
def test_invalid_json_request() -> None:
    """Test endpoint with invalid JSON"""
    response = client.post(
        "/auth/user/email",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
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
    with (
        patch("backend.api.features.v1.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.v1.get_cloud_storage_handler"
        ) as mock_handler_getter,
    ):
        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.return_value = "gcs://test-bucket/uploads/123/test.txt"
        mock_handler_getter.return_value = mock_handler

        # Mock file.read()
        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(
            file=upload_file_mock,
            user_id=test_user_id,
            ctx=_test_ctx(test_user_id),
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

    with (
        patch("backend.api.features.v1.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.v1.get_cloud_storage_handler"
        ) as mock_handler_getter,
    ):
        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.return_value = (
            "gcs://test-bucket/uploads/123/uploaded_file"
        )
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(
            file=upload_file_mock, user_id=test_user_id, ctx=_test_ctx(test_user_id)
        )

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
            file=upload_file_mock,
            user_id=test_user_id,
            ctx=_test_ctx(test_user_id),
            expiration_hours=0,
        )
    assert exc_info.value.status_code == 400
    assert "between 1 and 48" in exc_info.value.detail

    # Test expiration too long
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(
            file=upload_file_mock,
            user_id=test_user_id,
            ctx=_test_ctx(test_user_id),
            expiration_hours=49,
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

    with patch("backend.api.features.v1.scan_content_safe") as mock_scan:
        # Mock virus scan to raise exception
        mock_scan.side_effect = RuntimeError("Virus detected!")

        upload_file_mock.read = AsyncMock(return_value=file_content)

        with pytest.raises(RuntimeError, match="Virus detected!"):
            await upload_file(
                file=upload_file_mock,
                user_id=test_user_id,
                ctx=_test_ctx(test_user_id),
            )


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

    with (
        patch("backend.api.features.v1.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.v1.get_cloud_storage_handler"
        ) as mock_handler_getter,
    ):
        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.store_file.side_effect = RuntimeError("Storage error!")
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        with pytest.raises(RuntimeError, match="Storage error!"):
            await upload_file(
                file=upload_file_mock,
                user_id=test_user_id,
                ctx=_test_ctx(test_user_id),
            )


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
        await upload_file(
            file=upload_file_mock,
            user_id=test_user_id,
            ctx=_test_ctx(test_user_id),
        )

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

    with (
        patch("backend.api.features.v1.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.v1.get_cloud_storage_handler"
        ) as mock_handler_getter,
    ):
        mock_scan.return_value = None
        mock_handler = AsyncMock()
        mock_handler.config.gcs_bucket_name = ""  # Simulate no GCS bucket configured
        mock_handler_getter.return_value = mock_handler

        upload_file_mock.read = AsyncMock(return_value=file_content)

        result = await upload_file(
            file=upload_file_mock,
            user_id=test_user_id,
            ctx=_test_ctx(test_user_id),
        )

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
