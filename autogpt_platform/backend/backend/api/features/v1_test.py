import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import ANY, AsyncMock, Mock, patch

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
import starlette.datastructures
from fastapi import HTTPException, UploadFile
from pytest_snapshot.plugin import Snapshot

from backend.api.features.store.exceptions import VirusDetectedError
from backend.api.rest_api import handle_internal_http_error
from backend.copilot.tools.skills import (
    BuiltInSkillError,
    ParsedSkill,
    SkillLimitError,
    SkillNotFoundError,
)
from backend.data.credit import AutoTopUpConfig
from backend.data.graph import GraphModel
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import InsufficientBalanceError

from .v1 import upload_file, v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)
# Mirror rest_api.py's GraphActivationError → 400 mapping so the atomicity
# tests below verify the same behaviour the real app exposes.
app.add_exception_handler(GraphActivationError, handle_internal_http_error(400))

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
    mock_user.created_at = datetime.now(timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
        "name": "Test User",
    }

    mocker.patch(
        "backend.api.features.v1.get_or_create_user",
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
        "backend.api.features.v1.get_user_credit_model",
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


# Credits endpoints tests
def test_get_user_credits(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get user credits endpoint"""
    mock_credit_model = Mock()
    mock_credit_model.get_credits = AsyncMock(return_value=1000)
    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
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
        "backend.api.features.v1.get_user_credit_model",
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


def test_request_top_up_forwards_datafast_headers(
    mocker: pytest_mock.MockFixture,
) -> None:
    """DataFast attribution headers are forwarded to top_up_intent."""
    mock_credit_model = Mock()
    mock_credit_model.top_up_intent = AsyncMock(
        return_value="https://checkout.example.com/session123"
    )
    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.post(
        "/credits",
        json={"credit_amount": 500},
        headers={
            "X-Datafast-Visitor-Id": "vis_1",
            "X-Datafast-Session-Id": "ses_1",
        },
    )

    assert response.status_code == 200
    mock_credit_model.top_up_intent.assert_awaited_once_with(
        ANY,
        500,
        datafast_visitor_id="vis_1",
        datafast_session_id="ses_1",
    )


def test_get_auto_top_up(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get auto top-up configuration endpoint"""
    mock_config = AutoTopUpConfig(threshold=100, amount=500)

    mocker.patch(
        "backend.api.features.v1.get_auto_top_up",
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
        "backend.api.features.v1.set_auto_top_up",
        return_value=None,
    )

    # Mock credit model to avoid Stripe API calls
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50  # Current balance below threshold
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
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
    mocker.patch("backend.api.features.v1.set_auto_top_up")

    # Mock credit model to avoid Stripe API calls for the successful case
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
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


def test_list_invoices_returns_mapped_payload(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The /credits/invoices route should return whatever the credit model
    yields, serialised through the InvoiceListItem schema."""
    from backend.data.credit import InvoiceListItem

    invoice = InvoiceListItem(
        id="in_1",
        number="INV-001",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        total_cents=2500,
        amount_paid_cents=0,
        currency="usd",
        status="open",
        description="Subscription",
        hosted_invoice_url="https://invoice.stripe.com/i/test",
        invoice_pdf_url="https://invoice.stripe.com/i/test/pdf",
    )

    mock_credit_model = Mock()
    mock_credit_model.list_invoices = AsyncMock(return_value=[invoice])
    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.get("/credits/invoices?limit=24")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    row = payload[0]
    assert row["id"] == "in_1"
    assert row["total_cents"] == 2500
    assert row["amount_paid_cents"] == 0
    assert row["status"] == "open"
    assert row["hosted_invoice_url"] == "https://invoice.stripe.com/i/test"
    mock_credit_model.list_invoices.assert_awaited_once()
    # Ensure the limit query param is forwarded.
    assert mock_credit_model.list_invoices.await_args.kwargs == {"limit": 24}


def test_list_invoices_clamps_limit(mocker: pytest_mock.MockFixture) -> None:
    """FastAPI's Query(le=100) should reject limit > 100."""
    mock_credit_model = Mock()
    mock_credit_model.list_invoices = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.get("/credits/invoices?limit=500")

    assert response.status_code == 422  # Validation error
    mock_credit_model.list_invoices.assert_not_awaited()


def test_list_invoices_default_limit(mocker: pytest_mock.MockFixture) -> None:
    """Omitting ?limit should default to 24."""
    mock_credit_model = Mock()
    mock_credit_model.list_invoices = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.v1.get_user_credit_model",
        return_value=mock_credit_model,
    )

    response = client.get("/credits/invoices")

    assert response.status_code == 200
    assert response.json() == []
    assert mock_credit_model.list_invoices.await_args.kwargs == {"limit": 24}


# Executions cost summary tests
def test_executions_cost_summary_returns_payload(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The /executions/cost-summary route returns the aggregated payload."""
    from prisma.enums import AgentExecutionStatus

    from backend.data.execution_cost_summary import (
        UserAgentCostRollup,
        UserDailyCost,
        UserExecutionCostSummary,
        UserTopRun,
    )

    summary = UserExecutionCostSummary(
        total_cents=4200,
        run_count=12,
        billable_run_count=10,
        failed_cost_cents=500,
        by_agent=[
            UserAgentCostRollup(graph_id="g-1", cost_cents=3000, run_count=8),
            UserAgentCostRollup(graph_id="g-2", cost_cents=1200, run_count=4),
        ],
        top_runs=[
            UserTopRun(
                execution_id="exec-1",
                graph_id="g-1",
                cost_cents=2500,
                started_at=datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
                status=AgentExecutionStatus.COMPLETED,
                duration_seconds=45.5,
                node_error_count=0,
            ),
        ],
        daily=[
            UserDailyCost(date="2026-05-10", cost_cents=3000, run_count=8),
            UserDailyCost(date="2026-05-11", cost_cents=1200, run_count=4),
        ],
    )

    mock_fn = mocker.patch(
        "backend.api.features.v1.get_user_cost_summary",
        AsyncMock(return_value=summary),
    )

    response = client.get("/executions/cost-summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_cents"] == 4200
    assert payload["run_count"] == 12
    assert payload["billable_run_count"] == 10
    assert payload["failed_cost_cents"] == 500
    assert len(payload["by_agent"]) == 2
    assert payload["by_agent"][0]["graph_id"] == "g-1"
    assert payload["by_agent"][0]["cost_cents"] == 3000
    assert len(payload["top_runs"]) == 1
    assert payload["top_runs"][0]["execution_id"] == "exec-1"
    assert payload["top_runs"][0]["cost_cents"] == 2500
    assert len(payload["daily"]) == 2
    assert payload["daily"][0]["date"] == "2026-05-10"
    mock_fn.assert_awaited_once()


def test_executions_cost_summary_forwards_since_until(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """since/until query params should reach get_user_cost_summary."""
    from backend.data.execution_cost_summary import UserExecutionCostSummary

    mock_fn = mocker.patch(
        "backend.api.features.v1.get_user_cost_summary",
        AsyncMock(
            return_value=UserExecutionCostSummary(
                total_cents=0,
                run_count=0,
                billable_run_count=0,
                failed_cost_cents=0,
                by_agent=[],
                top_runs=[],
                daily=[],
            )
        ),
    )

    response = client.get(
        "/executions/cost-summary"
        "?since=2026-05-01T00:00:00Z"
        "&until=2026-05-15T00:00:00Z"
        "&top_runs_limit=5"
    )

    assert response.status_code == 200
    kwargs = mock_fn.await_args.kwargs
    assert kwargs["user_id"] == test_user_id
    assert kwargs["since"] == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert kwargs["until"] == datetime(2026, 5, 15, tzinfo=timezone.utc)
    assert kwargs["top_runs_limit"] == 5


def test_executions_cost_summary_rejects_out_of_range_limit(
    mocker: pytest_mock.MockFixture,
) -> None:
    """top_runs_limit must be within [1, 50]."""
    mock_fn = mocker.patch(
        "backend.api.features.v1.get_user_cost_summary",
        AsyncMock(),
    )

    response = client.get("/executions/cost-summary?top_runs_limit=500")

    assert response.status_code == 422
    mock_fn.assert_not_awaited()


def test_executions_cost_summary_rejects_inverted_window(
    mocker: pytest_mock.MockFixture,
) -> None:
    """`since > until` is bad client input — surface 422, don't quietly return zeros."""
    mock_fn = mocker.patch(
        "backend.api.features.v1.get_user_cost_summary",
        AsyncMock(),
    )

    response = client.get(
        "/executions/cost-summary"
        "?since=2026-05-15T00:00:00Z"
        "&until=2026-05-01T00:00:00Z"
    )

    assert response.status_code == 422
    mock_fn.assert_not_awaited()


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
        "backend.api.features.v1.graph_db.get_graph",
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
        "backend.api.features.v1.graph_db.get_graph",
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
        "backend.api.features.v1.graph_db.get_graph",
        return_value=mock_graph,
    )
    mocker.patch(
        "backend.api.features.v1.on_graph_deactivate",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.v1.graph_db.delete_graph",
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


def test_create_new_graph_returns_400_and_persists_nothing_on_activation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Core atomicity guarantee: when before_graph_activate raises,
    POST /graphs must return 400 and never call create_graph / create_library_agent.
    Reordering activation back to post-save would break this test."""
    from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

    mock_graph_model = Mock()
    mocker.patch(
        "backend.api.features.v1.graph_db.make_graph_model",
        return_value=mock_graph_model,
    )
    activate_mock = mocker.patch(
        "backend.api.features.v1.before_graph_activate",
        new=AsyncMock(
            side_effect=GraphActivationError(
                "Credential #cred-1 needs reconnect — please reconnect"
            )
        ),
    )
    create_graph_mock = mocker.patch(
        "backend.api.features.v1.graph_db.create_graph", new=AsyncMock()
    )
    create_lib_agent_mock = mocker.patch(
        "backend.api.features.v1.library_db.create_library_agent",
        new=AsyncMock(),
    )

    response = client.post(
        "/graphs",
        json={
            "graph": {
                "name": "Test Graph",
                "description": "Test",
                "nodes": [],
                "links": [],
            }
        },
    )

    assert response.status_code == 400
    assert "reconnect" in response.json()["detail"]
    activate_mock.assert_awaited_once()
    create_graph_mock.assert_not_awaited()
    create_lib_agent_mock.assert_not_awaited()


def test_update_graph_returns_400_and_persists_nothing_on_activation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Same atomicity guarantee on PUT /graphs/{id}: an activation failure
    must short-circuit with 400 before any new graph version is written."""
    from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

    mock_graph_model = Mock(is_active=True)
    existing_version = Mock(version=1, is_active=True)
    mocker.patch(
        "backend.api.features.v1.graph_db.get_graph_all_versions",
        new=AsyncMock(return_value=[existing_version]),
    )
    mocker.patch(
        "backend.api.features.v1.graph_db.make_graph_model",
        return_value=mock_graph_model,
    )
    activate_mock = mocker.patch(
        "backend.api.features.v1.before_graph_activate",
        new=AsyncMock(
            side_effect=GraphActivationError(
                "Credential #cred-1 needs reconnect — please reconnect"
            )
        ),
    )
    create_graph_mock = mocker.patch(
        "backend.api.features.v1.graph_db.create_graph", new=AsyncMock()
    )
    update_lib_agent_mock = mocker.patch(
        "backend.api.features.v1.library_db.update_library_agent_version_and_settings",
        new=AsyncMock(),
    )

    response = client.put(
        "/graphs/graph-123",
        json={
            "id": "graph-123",
            "name": "Test Graph",
            "description": "Test",
            "nodes": [],
            "links": [],
        },
    )

    assert response.status_code == 400
    assert "reconnect" in response.json()["detail"]
    activate_mock.assert_awaited_once()
    create_graph_mock.assert_not_awaited()
    update_lib_agent_mock.assert_not_awaited()


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

    with patch("backend.api.features.v1.scan_content_safe") as mock_scan:
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


def test_list_copilot_turn_schedules_filters_to_copilot_kind(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """GET /schedules/followups returns only CopilotTurnJobInfo items for the user.

    The route delegates to ``Scheduler.get_execution_schedules(kind="copilot_turn")``;
    we mock the client to make sure (a) the kind filter is forwarded and
    (b) any non-copilot rows are dropped from the response.
    """
    from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

    copilot_info = CopilotTurnJobInfo(
        id="sched-1",
        name="copilot followup",
        next_run_time="2026-05-22T10:00:00+00:00",
        timezone="UTC",
        user_id=test_user_id,
        session_id="sess-1",
        message="check status",
        cron="0 9 * * *",
    )
    graph_info = GraphExecutionJobInfo(
        id="sched-2",
        name="graph run",
        next_run_time="2026-05-22T11:00:00+00:00",
        timezone="UTC",
        user_id=test_user_id,
        graph_id="g-1",
        graph_version=1,
        cron="0 10 * * *",
        input_data={},
    )

    mock_client = Mock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[copilot_info, graph_info]
    )
    mocker.patch(
        "backend.api.features.v1.get_scheduler_client",
        return_value=mock_client,
    )

    response = client.get("/schedules/followups")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "sched-1"
    assert body[0]["kind"] == "copilot_turn"
    assert body[0]["session_id"] == "sess-1"

    mock_client.get_execution_schedules.assert_awaited_once_with(
        user_id=test_user_id, kind="copilot_turn"
    )


def test_list_copilot_skills_returns_user_skills(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """GET /skills returns user-distilled skills (defaults are excluded
    because the UI hides them).
    """

    mocker.patch(
        "backend.api.features.v1.list_user_skills",
        AsyncMock(
            return_value=[
                ParsedSkill(
                    name="oauth_flow",
                    description="OAuth handshake recipe",
                    body="...",
                    triggers=("auth", "oauth"),
                ),
                ParsedSkill(
                    name="zzz_cleanup",
                    description="Cleanup recipe",
                    body="...",
                ),
            ]
        ),
    )

    response = client.get("/skills")
    assert response.status_code == 200
    body = response.json()
    assert [s["name"] for s in body] == ["oauth_flow", "zzz_cleanup"]
    assert body[0]["triggers"] == ["auth", "oauth"]
    assert body[1]["triggers"] == []


def test_delete_copilot_skill_returns_name_on_success(
    mocker: pytest_mock.MockFixture,
) -> None:
    """DELETE /skills/{name} returns the slug and forwards the user_id."""
    delete_mock = AsyncMock(return_value="my_skill")
    mocker.patch("backend.api.features.v1.delete_user_skill", delete_mock)

    response = client.delete("/skills/my_skill")
    assert response.status_code == 200
    assert response.json() == {"name": "my_skill"}
    delete_mock.assert_awaited_once()


def test_delete_copilot_skill_rejects_builtin(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Built-in defaults must not be user-deletable via the REST endpoint."""

    mocker.patch(
        "backend.api.features.v1.delete_user_skill",
        AsyncMock(side_effect=BuiltInSkillError("built-in")),
    )

    response = client.delete("/skills/agent_building_guide")
    assert response.status_code == 400


def test_delete_copilot_skill_returns_404_when_missing(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Missing skills surface as 404 so the UI can reconcile its list."""

    mocker.patch(
        "backend.api.features.v1.delete_user_skill",
        AsyncMock(side_effect=SkillNotFoundError("gone")),
    )

    response = client.delete("/skills/missing")
    assert response.status_code == 404


def test_read_copilot_skill_returns_user_body(
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /skills/{name} returns the full SKILL.md body for a user skill."""

    mocker.patch(
        "backend.api.features.v1.read_user_skill_with_body",
        AsyncMock(
            return_value=ParsedSkill(
                name="oauth_flow",
                description="OAuth handshake recipe",
                body="# OAuth flow\n\nStep 1: ...",
                triggers=("auth",),
                version="1",
            )
        ),
    )

    response = client.get("/skills/oauth_flow")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "oauth_flow"
    assert body["body"].startswith("# OAuth flow")
    assert body["triggers"] == ["auth"]
    assert body["version"] == "1"
    assert body["is_default"] is False


def test_read_copilot_skill_returns_default_with_body(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A built-in default name returns is_default=True and a non-empty body."""

    mocker.patch(
        "backend.api.features.v1.get_default_skill_with_body",
        return_value=ParsedSkill(
            name="agent_building_guide",
            description="default desc",
            body="# Default body\n",
            triggers=("create_agent",),
        ),
    )

    response = client.get("/skills/agent_building_guide")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "agent_building_guide"
    assert body["is_default"] is True
    assert body["body"].startswith("# Default body")


def test_read_copilot_skill_returns_404_when_missing(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A user-skill slug that has no SKILL.md surfaces as 404."""
    mocker.patch(
        "backend.api.features.v1.read_user_skill_with_body",
        AsyncMock(return_value=None),
    )

    response = client.get("/skills/missing")
    assert response.status_code == 404


_VALID_SKILL_MD = (
    "---\n"
    "name: oauth_flow\n"
    "description: OAuth handshake recipe\n"
    "triggers:\n"
    "  - auth\n"
    "---\n\n"
    "# OAuth flow\n\nStep 1: redirect to /authorize\n"
)


def test_upload_copilot_skill_creates_skill(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /skills parses the SKILL.md and persists it via store_user_skill."""
    store_mock = AsyncMock(
        return_value=ParsedSkill(
            name="oauth_flow",
            description="OAuth handshake recipe",
            body="# OAuth flow",
            triggers=("auth",),
        )
    )
    mocker.patch("backend.api.features.v1.store_user_skill", store_mock)

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "oauth_flow"
    assert body["triggers"] == ["auth"]
    store_mock.assert_awaited_once()
    assert store_mock.await_args.kwargs["name"] == "oauth_flow"


def test_upload_copilot_skill_rejects_malformed_markdown() -> None:
    """A file without valid frontmatter returns 400 before touching storage."""
    response = client.post(
        "/skills", json={"content": "just some text, no frontmatter"}
    )
    assert response.status_code == 400


def test_upload_copilot_skill_returns_409_when_at_cap(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The per-user cap surfaces as 409 so the UI can prompt a delete."""
    mocker.patch(
        "backend.api.features.v1.store_user_skill",
        AsyncMock(side_effect=SkillLimitError("Skill limit reached (50).")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 409


def test_upload_copilot_skill_returns_400_on_validation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A validation failure from store_user_skill maps to 400."""
    mocker.patch(
        "backend.api.features.v1.store_user_skill",
        AsyncMock(side_effect=ValueError("name must be a slug")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 400


def test_upload_copilot_skill_returns_400_on_virus_detection(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A virus-scan rejection surfaces as a 400 client error, not a 500."""
    mocker.patch(
        "backend.api.features.v1.store_user_skill",
        AsyncMock(side_effect=VirusDetectedError("nasty")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 400
    assert "virus scan" in response.json()["detail"]
