from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.features.v1 import get_credit_history, v1_router
from backend.api.rest_api import handle_internal_http_error
from backend.data.credit import UserCreditBase
from backend.data.model import TransactionHistory

CONTEXT = RequestContext(
    user_id="user",
    org_id="org",
    team_id=None,
    is_org_owner=True,
    is_org_admin=False,
    is_org_billing_manager=False,
    is_team_admin=False,
    is_team_billing_manager=False,
    seat_status="ACTIVE",
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(v1_router)
    app.add_exception_handler(ValueError, handle_internal_http_error(400))
    app.dependency_overrides[requires_user] = lambda: None
    app.dependency_overrides[get_user_id] = lambda: "user"
    app.dependency_overrides[get_request_context] = lambda: CONTEXT
    return TestClient(app)


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_invalid_history_limit_returns_400(client, limit):
    response = client.get(
        "/credits/transactions", params={"transaction_count_limit": limit}
    )
    assert response.status_code == 400
    assert "Transaction count limit" in response.json()["detail"]
    assert response.json()["message"] == "Failed to process GET /credits/transactions"


def test_invalid_cursor_returns_400(client, monkeypatch):
    model = AsyncMock(spec=UserCreditBase)
    model.get_transaction_history.side_effect = ValueError(
        "Invalid credit history cursor"
    )
    monkeypatch.setattr(
        "backend.api.features.v1.get_credit_model", AsyncMock(return_value=model)
    )
    response = client.get("/credits/transactions", params={"cursor": "invalid"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid credit history cursor"
    assert response.json()["message"] == "Failed to process GET /credits/transactions"


@pytest.mark.asyncio
async def test_history_route_forwards_cursor_and_org_context(monkeypatch):
    model = AsyncMock(spec=UserCreditBase)
    model.get_transaction_history.return_value = TransactionHistory(
        transactions=[], next_transaction_time=None
    )
    get_model = AsyncMock(return_value=model)
    monkeypatch.setattr("backend.api.features.v1.get_credit_model", get_model)
    result = await get_credit_history(
        user_id="user", ctx=CONTEXT, cursor="cursor", transaction_count_limit=50
    )
    get_model.assert_awaited_once_with("user", "org")
    model.get_transaction_history.assert_awaited_once_with(
        user_id="user",
        transaction_time_ceiling=None,
        transaction_count_limit=50,
        transaction_type=None,
        cursor="cursor",
        viewer_organization_id="org",
    )
    assert result.transactions == []
