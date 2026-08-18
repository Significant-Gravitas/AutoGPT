import json
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import prisma.enums
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pytest_snapshot.plugin import Snapshot

from backend.data.model import UserTransaction
from backend.util.json import SafeJson
from backend.util.models import Pagination

from .credit_admin_routes import router as credit_admin_router
from .model import UserHistoryResponse

app = fastapi.FastAPI()
app.include_router(credit_admin_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_add_user_credits_success(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
    admin_user_id: str,
    target_user_id: str,
) -> None:
    """Test successful credit addition by admin"""
    # Mock the credit model
    mock_credit_model = Mock()
    mock_credit_model._add_transaction = AsyncMock(
        return_value=(1500, "transaction-123-uuid")
    )
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.get_user_credit_model",
        return_value=mock_credit_model,
    )

    request_data = {
        "user_id": target_user_id,
        "amount": 500,
        "comments": "Test credit grant for debugging",
    }

    response = client.post("/admin/add_credits", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["new_balance"] == 1500
    assert response_data["transaction_key"] == "transaction-123-uuid"

    # Verify the function was called with correct parameters
    mock_credit_model._add_transaction.assert_called_once()
    call_args = mock_credit_model._add_transaction.call_args
    assert call_args[0] == (target_user_id, 500)
    assert call_args[1]["transaction_type"] == prisma.enums.CreditTransactionType.GRANT
    # Check that metadata is a SafeJson object with the expected content
    assert isinstance(call_args[1]["metadata"], SafeJson)
    actual_metadata = call_args[1]["metadata"]
    expected_data = {
        "admin_id": admin_user_id,
        "reason": "Test credit grant for debugging",
    }

    # SafeJson inherits from Json which stores parsed data in the .data attribute
    assert actual_metadata.data["admin_id"] == expected_data["admin_id"]
    assert actual_metadata.data["reason"] == expected_data["reason"]

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "admin_add_credits_success",
    )


def test_add_user_credits_negative_amount(
    mocker: pytest_mock.MockerFixture,
    snapshot: Snapshot,
) -> None:
    """Test credit deduction by admin (negative amount)"""
    # Mock the credit model
    mock_credit_model = Mock()
    mock_credit_model._add_transaction = AsyncMock(
        return_value=(200, "transaction-456-uuid")
    )
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.get_user_credit_model",
        return_value=mock_credit_model,
    )

    request_data = {
        "user_id": "target-user-id",
        "amount": -100,
        "comments": "Refund adjustment",
    }

    response = client.post("/admin/add_credits", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["new_balance"] == 200

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "adm_add_cred_neg",
    )


def test_get_user_history_success(
    mocker: pytest_mock.MockerFixture,
    snapshot: Snapshot,
) -> None:
    """Test successful retrieval of user credit history"""
    # Mock the admin_get_user_history function
    mock_history_response = UserHistoryResponse(
        history=[
            UserTransaction(
                user_id="user-1",
                user_email="user1@example.com",
                amount=1000,
                reason="Initial grant",
                transaction_type=prisma.enums.CreditTransactionType.GRANT,
            ),
            UserTransaction(
                user_id="user-2",
                user_email="user2@example.com",
                amount=-50,
                reason="Usage",
                transaction_type=prisma.enums.CreditTransactionType.USAGE,
            ),
        ],
        pagination=Pagination(
            total_items=2,
            total_pages=1,
            current_page=1,
            page_size=20,
        ),
    )

    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.admin_get_user_history",
        return_value=mock_history_response,
    )

    response = client.get("/admin/users_history")

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["history"]) == 2
    assert response_data["pagination"]["total_items"] == 2

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "adm_usr_hist_ok",
    )


def test_get_user_history_with_filters(
    mocker: pytest_mock.MockerFixture,
    snapshot: Snapshot,
) -> None:
    """Test user credit history with search and filter parameters"""
    # Mock the admin_get_user_history function
    mock_history_response = UserHistoryResponse(
        history=[
            UserTransaction(
                user_id="user-3",
                user_email="test@example.com",
                amount=500,
                reason="Top up",
                transaction_type=prisma.enums.CreditTransactionType.TOP_UP,
            ),
        ],
        pagination=Pagination(
            total_items=1,
            total_pages=1,
            current_page=1,
            page_size=10,
        ),
    )

    mock_get_history = mocker.patch(
        "backend.api.features.admin.credit_admin_routes.admin_get_user_history",
        return_value=mock_history_response,
    )

    response = client.get(
        "/admin/users_history",
        params={
            "search": "test@example.com",
            "page": 1,
            "page_size": 10,
            "transaction_filter": "TOP_UP",
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["history"]) == 1
    assert response_data["history"][0]["transaction_type"] == "TOP_UP"

    # Verify the function was called with correct parameters
    mock_get_history.assert_called_once_with(
        page=1,
        page_size=10,
        search="test@example.com",
        transaction_filter=prisma.enums.CreditTransactionType.TOP_UP,
        include_inactive=False,
    )

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "adm_usr_hist_filt",
    )


def test_get_user_history_empty_results(
    mocker: pytest_mock.MockerFixture,
    snapshot: Snapshot,
) -> None:
    """Test user credit history with no results"""
    # Mock empty history response
    mock_history_response = UserHistoryResponse(
        history=[],
        pagination=Pagination(
            total_items=0,
            total_pages=0,
            current_page=1,
            page_size=20,
        ),
    )

    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.admin_get_user_history",
        return_value=mock_history_response,
    )

    response = client.get("/admin/users_history", params={"search": "nonexistent"})

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["history"]) == 0
    assert response_data["pagination"]["total_items"] == 0

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "adm_usr_hist_empty",
    )


def test_add_credits_invalid_request() -> None:
    """Test credit addition with invalid request data"""
    # Missing required fields
    response = client.post("/admin/add_credits", json={})
    assert response.status_code == 422

    # Invalid amount type
    response = client.post(
        "/admin/add_credits",
        json={
            "user_id": "test",
            "amount": "not_a_number",
            "comments": "test",
        },
    )
    assert response.status_code == 422

    # Missing comments
    response = client.post(
        "/admin/add_credits",
        json={
            "user_id": "test",
            "amount": 100,
        },
    )
    assert response.status_code == 422


def test_admin_endpoints_require_admin_role(mock_jwt_user) -> None:
    """Test that admin endpoints require admin role"""
    # Simulate regular non-admin user
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # Test add_credits endpoint
    response = client.post(
        "/admin/add_credits",
        json={
            "user_id": "test",
            "amount": 100,
            "comments": "test",
        },
    )
    assert response.status_code == 403

    # Test users_history endpoint
    response = client.get("/admin/users_history")
    assert response.status_code == 403

    # Test new export endpoints
    response = client.get(
        "/admin/transactions/export",
        params={"start": "2026-01-01T00:00:00", "end": "2026-01-31T00:00:00"},
    )
    assert response.status_code == 403
    response = client.get(
        "/admin/copilot-usage/export",
        params={"start": "2026-01-01T00:00:00", "end": "2026-01-31T00:00:00"},
    )
    assert response.status_code == 403


def test_export_credit_transactions_success(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_export = AsyncMock(
        return_value=[
            UserTransaction(
                user_id="user-1",
                user_email="user1@example.com",
                amount=1000,
                running_balance=2500,
                reason="Initial grant",
                admin_email="admin@example.com",
                transaction_type=prisma.enums.CreditTransactionType.GRANT,
            ),
            UserTransaction(
                user_id="user-2",
                user_email="user2@example.com",
                amount=-50,
                running_balance=4500,
                reason="",
                transaction_type=prisma.enums.CreditTransactionType.USAGE,
            ),
        ]
    )
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.admin_export_user_history",
        mock_export,
    )

    response = client.get(
        "/admin/transactions/export",
        params={
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-31T00:00:00",
            "transaction_type": "GRANT",
            "user_id": "user-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 2
    assert body["window_days"] == 30
    assert body["max_window_days"] == 90
    assert body["transactions"][0]["user_email"] == "user1@example.com"

    call_kwargs = mock_export.call_args.kwargs
    assert call_kwargs["transaction_type"] == prisma.enums.CreditTransactionType.GRANT
    assert call_kwargs["user_id"] == "user-1"


def test_export_credit_transactions_window_too_wide_returns_400(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.admin_export_user_history",
        AsyncMock(side_effect=ValueError("Export window must be <= 90 days")),
    )
    response = client.get(
        "/admin/transactions/export",
        params={
            "start": "2025-01-01T00:00:00",
            "end": "2026-01-01T00:00:00",
        },
    )
    assert response.status_code == 400
    assert "90 days" in response.json()["detail"]


def test_export_credit_transactions_missing_window_is_400() -> None:
    response = client.get("/admin/transactions/export")
    assert response.status_code == 400


def test_export_copilot_weekly_usage_success(
    mocker: pytest_mock.MockerFixture,
) -> None:
    from datetime import datetime, timezone

    from backend.data.platform_cost import CopilotWeeklyUsageRow

    rows = [
        CopilotWeeklyUsageRow(
            user_id="user-1",
            user_email="user1@example.com",
            week_start=datetime(2026, 1, 5, tzinfo=timezone.utc),
            week_end=datetime(2026, 1, 12, tzinfo=timezone.utc),
            copilot_cost_microdollars=2_500_000,
            tier="PRO",
            weekly_limit_microdollars=25_000_000,
            percent_used=10.0,
        ),
    ]
    mock_export = AsyncMock(return_value=rows)
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.get_copilot_weekly_usage_for_export",
        mock_export,
    )

    response = client.get(
        "/admin/copilot-usage/export",
        params={
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-31T00:00:00",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 1
    assert body["max_window_days"] == 90
    row = body["rows"][0]
    assert row["user_id"] == "user-1"
    assert row["tier"] == "PRO"
    assert row["weekly_limit_microdollars"] == 25_000_000
    assert row["percent_used"] == 10.0


def test_export_copilot_weekly_usage_window_too_wide_returns_400(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.admin.credit_admin_routes.get_copilot_weekly_usage_for_export",
        AsyncMock(side_effect=ValueError("Export window must be <= 90 days")),
    )
    response = client.get(
        "/admin/copilot-usage/export",
        params={
            "start": "2025-01-01T00:00:00",
            "end": "2026-01-01T00:00:00",
        },
    )
    assert response.status_code == 400
