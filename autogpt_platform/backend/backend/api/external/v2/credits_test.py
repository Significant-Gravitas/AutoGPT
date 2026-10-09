import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
import stripe
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.data.model import TransactionHistory

from .credits import (
    _get_stripe_price_amount,
    get_balance,
    get_cost_summary,
    get_transactions,
    list_invoices,
)
from .pagination import PageRequest
from .tenancy import TenantContext, require_billing_permission


async def test_a_stripe_error_is_not_cached_as_a_zero_price(
    mocker: pytest_mock.MockFixture,
) -> None:
    _get_stripe_price_amount.cache_clear()
    mocker.patch.object(
        stripe.Price,
        "retrieve",
        side_effect=[stripe.StripeError("network"), SimpleNamespace(unit_amount=2000)],
    )

    await _get_stripe_price_amount("price_pro")
    assert await _get_stripe_price_amount("price_pro") == 2000


_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)


@pytest.fixture
def credit_model(mocker: pytest_mock.MockFixture) -> Mock:
    model = Mock(
        get_transaction_history=AsyncMock(
            return_value=TransactionHistory(
                transactions=[],
                next_transaction_time=datetime(2026, 10, 1, tzinfo=timezone.utc),
                next_cursor="history-cursor-2",
            )
        )
    )
    mocker.patch(
        "backend.api.external.v2.credits.get_credit_model",
        new_callable=AsyncMock,
        return_value=model,
    )
    return model


async def test_transactions_page_by_the_historys_own_cursor(credit_model: Mock):
    """A time ceiling alone skipped groups that tied on time at a page boundary."""
    first = await get_transactions(
        transaction_type=None, page=PageRequest(limit=10), auth=_AUTH
    )
    assert first.next_cursor is not None

    await get_transactions(
        transaction_type=None,
        page=PageRequest(limit=10, cursor=first.next_cursor),
        auth=_AUTH,
    )

    kwargs = credit_model.get_transaction_history.await_args.kwargs
    assert kwargs["cursor"] == "history-cursor-2"
    assert kwargs["viewer_organization_id"] == "org-1"
    assert "transaction_time_ceiling" not in kwargs


@pytest.mark.parametrize(
    "member, allowed",
    [
        (Mock(isOwner=True, isBillingManager=False), True),
        (Mock(isOwner=False, isBillingManager=True), True),
        (Mock(isOwner=False, isBillingManager=False), False),
        (None, False),
    ],
)
async def test_only_the_owner_or_a_billing_manager_reads_the_orgs_credits(
    mocker: pytest_mock.MockFixture, member: Mock | None, allowed: bool
):
    """The web app shows a pooled wallet only to those roles; keys follow suit."""
    prisma = mocker.patch("backend.api.external.v2.tenancy.prisma")
    prisma.orgmember.find_unique = AsyncMock(return_value=member)
    check = require_billing_permission(APIKeyPermission.READ_CREDITS)

    if allowed:
        assert await check(tenant=_AUTH) is _AUTH
    else:
        with pytest.raises(HTTPException) as raised:
            await check(tenant=_AUTH)
        assert raised.value.status_code == 403


def test_the_org_wallet_routes_check_the_billing_role():
    for route in (get_balance, get_transactions, list_invoices):
        source = inspect.getsource(route)
        assert "require_billing_permission" in source, route.__name__


async def test_a_cost_summary_takes_a_time_without_an_offset_as_utc(
    mocker: pytest_mock.MockFixture,
):
    summary = mocker.patch(
        "backend.api.external.v2.credits.get_user_cost_summary",
        new_callable=AsyncMock,
    )
    mocker.patch("backend.api.external.v2.credits.AutomationCostSummary.from_internal")

    await get_cost_summary(
        since=datetime(2026, 10, 1),
        until=datetime(2026, 10, 2, tzinfo=timezone.utc),
        top_runs_limit=10,
        auth=_AUTH,
    )

    assert summary.await_args.kwargs["since"] == datetime(
        2026, 10, 1, tzinfo=timezone.utc
    )
