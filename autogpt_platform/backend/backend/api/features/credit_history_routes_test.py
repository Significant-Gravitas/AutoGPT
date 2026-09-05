from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException

from backend.api.features.v1 import get_credit_history
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


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1001])
async def test_invalid_history_limit_returns_400(limit):
    with pytest.raises(HTTPException) as error:
        await get_credit_history(
            user_id="user", ctx=CONTEXT, transaction_count_limit=limit
        )
    assert error.value.status_code == 400
    assert "Transaction count limit" in error.value.detail


@pytest.mark.asyncio
async def test_invalid_cursor_returns_400(monkeypatch):
    model = AsyncMock(spec=UserCreditBase)
    model.get_transaction_history.side_effect = ValueError(
        "Invalid credit history cursor"
    )
    monkeypatch.setattr(
        "backend.api.features.v1.get_credit_model", AsyncMock(return_value=model)
    )
    with pytest.raises(HTTPException) as error:
        await get_credit_history(user_id="user", ctx=CONTEXT, cursor="invalid")
    assert error.value.status_code == 400
    assert error.value.detail == "Invalid credit history cursor"


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
