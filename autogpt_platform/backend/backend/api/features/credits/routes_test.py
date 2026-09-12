"""Tests for the credit ledger routes."""

import json
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute
from pytest_snapshot.plugin import Snapshot

from backend.api.features.credits.routes import router
from backend.api.rest_api import app as real_app
from backend.data.credit import AutoTopUpConfig

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # The real get_request_context queries Prisma to resolve the personal org,
    # which closes the test event loop between sync TestClient calls.
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


EXPECTED_OPERATIONS = {
    ("get", "/api/credits"),
    ("post", "/api/credits"),
    ("patch", "/api/credits"),
    ("post", "/api/credits/{transaction_key}/refund"),
    ("get", "/api/credits/auto-top-up"),
    ("post", "/api/credits/auto-top-up"),
    ("get", "/api/credits/transactions"),
    ("get", "/api/credits/refunds"),
    ("get", "/api/credits/invoices"),
}


@pytest.mark.parametrize("method,path", sorted(EXPECTED_OPERATIONS))
def test_credit_operation_is_published(method: str, path: str):
    operation = real_app.openapi()["paths"][path][method]
    assert operation["tags"] == ["v1", "credits"]
    assert "security" in operation


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_credit_route_requires_an_authenticated_user(path: str):
    """`security` in the schema does not prove this: each handler's own
    Security(get_user_id) puts it there, so removing the router's requires_user
    leaves the schema unchanged. Assert the dependency."""
    for route in real_app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            assert "requires_user" in {
                d.call.__name__ for d in route.dependant.dependencies if d.call
            }


def test_credit_surface_has_no_other_operations():
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.credits.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == EXPECTED_OPERATIONS


# The subscription and Stripe routes share the /credits prefix from their own
# module; /credits/{transaction_key}/refund is the only wildcard among them.
@pytest.mark.parametrize(
    "path,module",
    [
        (
            "/api/credits/subscription",
            "backend.api.features.subscriptions.routes",
        ),
        (
            "/api/credits/manage",
            "backend.api.features.subscriptions.routes",
        ),
        (
            "/api/credits/stripe_webhook",
            "backend.api.features.subscriptions.routes",
        ),
        ("/api/credits/transactions", "backend.api.features.credits.routes"),
        (
            "/api/credits/{transaction_key}/refund",
            "backend.api.features.credits.routes",
        ),
    ],
)
def test_credit_subpath_is_served_by_the_expected_module(path: str, module: str):
    handlers = {
        route.endpoint.__module__
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {module}


def test_get_user_credits(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test get user credits endpoint"""
    mock_credit_model = Mock()
    mock_credit_model.get_credits = AsyncMock(return_value=1000)
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_auto_top_up",
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
        "backend.api.features.credits.routes.set_auto_top_up",
        return_value=None,
    )

    # Mock credit model to avoid Stripe API calls
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50  # Current balance below threshold
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
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


def test_configure_auto_top_up_refuses_when_model_has_no_payment_path(
    mocker: pytest_mock.MockFixture,
) -> None:
    set_auto_top_up = mocker.patch(
        "backend.api.features.credits.routes.set_auto_top_up"
    )

    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50
    mock_credit_model.top_up_credits.side_effect = NotImplementedError(
        "Org-level Stripe top-up not yet implemented"
    )
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
        return_value=mock_credit_model,
    )

    response = client.post(
        "/credits/auto-top-up", json={"threshold": 500, "amount": 500}
    )

    assert response.status_code == 501
    set_auto_top_up.assert_not_called()


def test_configure_auto_top_up_validation_errors(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test configure auto top-up endpoint validation"""
    # Mock set_auto_top_up to avoid database operations for successful case
    mocker.patch("backend.api.features.credits.routes.set_auto_top_up")

    # Mock credit model to avoid Stripe API calls for the successful case
    mock_credit_model = mocker.AsyncMock()
    mock_credit_model.get_credits.return_value = 50
    mock_credit_model.top_up_credits.return_value = None

    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_credit_model",
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
        "backend.api.features.credits.routes.get_credit_model",
        return_value=mock_credit_model,
    )

    response = client.get("/credits/invoices")

    assert response.status_code == 200
    assert response.json() == []
    assert mock_credit_model.list_invoices.await_args.kwargs == {"limit": 24}


def test_refund_top_up_forwards_the_transaction_key_and_metadata(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The key identifies which top-up is being refunded; dropping it would
    refund the wrong transaction."""
    credit_model = Mock()
    credit_model.top_up_refund = AsyncMock(return_value=500)
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )

    response = client.post("/credits/txn-1/refund", json={"reason": "duplicate"})

    assert response.status_code == 200
    args = credit_model.top_up_refund.await_args.args
    assert args[1] == "txn-1"
    assert args[2] == {"reason": "duplicate"}


def test_fulfill_checkout_credits_the_calling_user(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Fulfilment is keyed on the authenticated user, not on anything in the
    request — that is what stops one account fulfilling another's checkout."""
    credit_model = Mock()
    credit_model.fulfill_checkout = AsyncMock()
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )

    response = client.patch("/credits")

    assert response.status_code == 200
    assert set(credit_model.fulfill_checkout.await_args.kwargs) == {"user_id"}


def test_get_refund_requests_is_scoped_to_the_caller(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    credit_model = Mock()
    credit_model.get_refund_requests = AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )

    response = client.get("/credits/refunds")

    assert response.status_code == 200
    assert credit_model.get_refund_requests.await_args.args[0] == test_user_id


def test_configure_auto_top_up_maps_an_unavailable_backend_to_501(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A context without auto top-up must say so, not surface as a 500."""
    credit_model = Mock()
    credit_model.get_credits = AsyncMock(return_value=0)
    credit_model.top_up_credits = AsyncMock(side_effect=NotImplementedError)
    mocker.patch(
        "backend.api.features.credits.routes.get_credit_model",
        AsyncMock(return_value=credit_model),
    )

    response = client.post(
        "/credits/auto-top-up", json={"threshold": 1000, "amount": 1000}
    )

    assert response.status_code == 501


def test_request_top_up_rejects_a_missing_amount() -> None:
    """Moved with its route from v1_test's invalid-request section."""
    response = client.post("/credits", json={})

    assert response.status_code == 422
