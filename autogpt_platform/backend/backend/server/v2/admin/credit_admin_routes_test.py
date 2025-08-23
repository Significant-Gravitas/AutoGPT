import json
from unittest.mock import AsyncMock

import autogpt_libs.auth
import autogpt_libs.auth.depends
import fastapi
import fastapi.testclient
import prisma.enums
import pytest_mock
from prisma import Json
from pytest_snapshot.plugin import Snapshot

import backend.server.v2.admin.credit_admin_routes as credit_admin_routes
import backend.server.v2.admin.model as admin_model
from backend.data.model import UserTransaction
from backend.server.conftest import ADMIN_USER_ID, TARGET_USER_ID
from backend.util.models import Pagination

app = fastapi.FastAPI()
app.include_router(credit_admin_routes.router)

client = fastapi.testclient.TestClient(app)


def override_requires_admin_user() -> dict[str, str]:
    """Override admin user check for testing"""
    return {"sub": ADMIN_USER_ID, "role": "admin"}


def override_get_user_id() -> str:
    """Override get_user_id for testing"""
    return ADMIN_USER_ID


app.dependency_overrides[autogpt_libs.auth.requires_admin_user] = (
    override_requires_admin_user
)
app.dependency_overrides[autogpt_libs.auth.depends.get_user_id] = override_get_user_id


def test_add_user_credits_success(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
) -> None:
    """Test successful credit addition by admin"""
    # Mock the credit model
    mock_credit_model = mocker.patch(
        "backend.server.v2.admin.credit_admin_routes._user_credit_model"
    )
    mock_credit_model._add_transaction = AsyncMock(
        return_value=(1500, "transaction-123-uuid")
    )

    request_data = {
        "user_id": TARGET_USER_ID,
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
    assert call_args[0] == (TARGET_USER_ID, 500)
    assert call_args[1]["transaction_type"] == prisma.enums.CreditTransactionType.GRANT
    # Check that metadata is a Json object with the expected content
    assert isinstance(call_args[1]["metadata"], Json)
    assert call_args[1]["metadata"] == Json(
        {"admin_id": ADMIN_USER_ID, "reason": "Test credit grant for debugging"}
    )

    # Snapshot test the response
    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "admin_add_credits_success",
    )


def test_add_user_credits_negative_amount(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test credit deduction by admin (negative amount)"""
    # Mock the credit model
    mock_credit_model = mocker.patch(
        "backend.server.v2.admin.credit_admin_routes._user_credit_model"
    )
    mock_credit_model._add_transaction = AsyncMock(
        return_value=(200, "transaction-456-uuid")
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
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test successful retrieval of user credit history"""
    # Mock the admin_get_user_history function
    mock_history_response = admin_model.UserHistoryResponse(
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
        "backend.server.v2.admin.credit_admin_routes.admin_get_user_history",
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
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test user credit history with search and filter parameters"""
    # Mock the admin_get_user_history function
    mock_history_response = admin_model.UserHistoryResponse(
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
        "backend.server.v2.admin.credit_admin_routes.admin_get_user_history",
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
    )

    # Snapshot test the response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "adm_usr_hist_filt",
    )


def test_get_user_history_empty_results(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test user credit history with no results"""
    # Mock empty history response
    mock_history_response = admin_model.UserHistoryResponse(
        history=[],
        pagination=Pagination(
            total_items=0,
            total_pages=0,
            current_page=1,
            page_size=20,
        ),
    )

    mocker.patch(
        "backend.server.v2.admin.credit_admin_routes.admin_get_user_history",
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


def test_admin_endpoints_require_admin_role(mocker: pytest_mock.MockFixture) -> None:
    """Test that admin endpoints require admin role"""
    # Clear the admin override to test authorization
    app.dependency_overrides.clear()

    # Mock requires_admin_user to raise an exception
    mocker.patch(
        "autogpt_libs.auth.requires_admin_user",
        side_effect=fastapi.HTTPException(
            status_code=403, detail="Admin access required"
        ),
    )

    # Test add_credits endpoint
    response = client.post(
        "/admin/add_credits",
        json={
            "user_id": "test",
            "amount": 100,
            "comments": "test",
        },
    )
    assert (
        response.status_code == 401
    )  # Auth middleware returns 401 when auth is disabled

    # Test users_history endpoint
    response = client.get("/admin/users_history")
    assert (
        response.status_code == 401
    )  # Auth middleware returns 401 when auth is disabled

    # Restore the override
    app.dependency_overrides[autogpt_libs.auth.requires_admin_user] = (
        override_requires_admin_user
    )
    app.dependency_overrides[autogpt_libs.auth.depends.get_user_id] = (
        override_get_user_id
    )
