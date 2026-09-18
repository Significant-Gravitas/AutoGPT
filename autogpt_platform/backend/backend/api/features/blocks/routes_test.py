"""Tests for the block and file-upload routes."""

import json
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
import starlette.datastructures
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException, UploadFile
from fastapi.routing import APIRoute
from pytest_snapshot.plugin import Snapshot

from backend.api.features.blocks.routes import router, upload_file
from backend.api.rest_api import app as real_app
from backend.util.exceptions import InsufficientBalanceError


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
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

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


# Tags differ across the three, so they stay per-route and the mount supplies
# only ["v1"]; execute keeps its paywall on top of the router's auth.
EXPECTED_OPERATIONS = {
    ("get", "/api/blocks"): ["v1", "blocks"],
    ("post", "/api/blocks/{block_id}/execute"): ["v1", "blocks"],
    ("post", "/api/files/upload"): ["v1", "files"],
}


@pytest.mark.parametrize(
    "method,path,tags", [(m, p, t) for (m, p), t in EXPECTED_OPERATIONS.items()]
)
def test_block_operation_is_published(method: str, path: str, tags: list[str]):
    operation = real_app.openapi()["paths"][path][method]
    assert operation["tags"] == tags


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_block_route_requires_an_authenticated_user(path: str):
    """`security` in the schema does not prove this — each handler's own
    Security(get_user_id) puts it there. Assert the dependency."""
    for route in real_app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            assert "requires_user" in {
                d.call.__name__ for d in route.dependant.dependencies if d.call
            }


def test_execute_block_is_behind_the_payment_paywall():
    """The only one of the three with a dependency beyond auth, and it gates
    spending."""
    route = next(
        r
        for r in real_app.routes
        if isinstance(r, APIRoute) and r.path == "/api/blocks/{block_id}/execute"
    )
    assert "enforce_payment_paywall" in {
        d.call.__name__ for d in route.dependant.dependencies if d.call
    }


def test_block_surface_has_no_other_operations():
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.blocks.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == set(EXPECTED_OPERATIONS)


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
        "backend.api.features.blocks.routes.get_blocks",
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
        "backend.api.features.blocks.routes.get_block",
        return_value=mock_block,
    )

    # Mock user for user_context
    mock_user = Mock()
    mock_user.timezone = "UTC"

    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
        return_value=mock_user,
    )

    # Default to free block: cost = 0, no charge call. The credit model is
    # patched where the charge would actually go through — the sibling
    # charging test below patches the same target — so "not awaited" is a
    # claim about the code path rather than about an unrelated mock.
    cost_mock = mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.block_usage_cost",
        return_value=(0, {}),
    )
    mock_credit_model = mocker.AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_user_credit_model",
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
        "backend.api.features.blocks.routes.get_block",
        return_value=mock_block,
    )

    mock_user = Mock()
    mock_user.timezone = "America/New_York"
    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
        return_value=mock_user,
    )

    mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.block_usage_cost",
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
        "backend.api.features.blocks.routes.get_block",
        return_value=mock_block,
    )
    mock_user = Mock()
    mock_user.timezone = "UTC"
    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
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
        "backend.api.features.blocks.routes.get_block",
        return_value=mock_block,
    )
    mock_user = Mock()
    mock_user.timezone = "UTC"
    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
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
        "backend.api.features.blocks.routes.get_block",
        return_value=None,
    )

    response = client.post("/blocks/nonexistent-block/execute", json={})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


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
        patch("backend.api.features.blocks.routes.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.blocks.routes.get_cloud_storage_handler"
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
        patch("backend.api.features.blocks.routes.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.blocks.routes.get_cloud_storage_handler"
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

    with patch("backend.api.features.blocks.routes.scan_content_safe") as mock_scan:
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
        patch("backend.api.features.blocks.routes.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.blocks.routes.get_cloud_storage_handler"
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
        patch("backend.api.features.blocks.routes.scan_content_safe") as mock_scan,
        patch(
            "backend.api.features.blocks.routes.get_cloud_storage_handler"
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
