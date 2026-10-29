"""Regression guards for the single-ledger invariant.

Executions now carry org/team on their ExecutionContext, which routes
billing through spend_org_credits. Personal orgs MUST bill the owner's
user wallet — Stripe top-ups, grants, refunds, and auto-top-up all land
on the user ledger, so debiting the migration-seeded OrgBalance copy
would drift the ledgers: a paying user's runs would fail on a stale org
balance no payment can replenish.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import org_credit
from backend.data.org_credit import (
    get_org_credits,
    spend_org_credits,
    top_up_org_credits,
)

ORG_PERSONAL = "org-personal"
ORG_TEAM = "org-team"
OWNER = "owner-user"


@pytest.fixture(autouse=True)
def clear_owner_cache():
    org_credit._personal_org_owner.cache_clear()
    yield
    org_credit._personal_org_owner.cache_clear()


@pytest.fixture
def mock_prisma(mocker):
    p = MagicMock()

    async def find_org(where):
        org = MagicMock()
        org.id = where["id"]
        org.isPersonal = where["id"] == ORG_PERSONAL
        return org

    p.organization.find_unique = AsyncMock(side_effect=find_org)
    owner_member = MagicMock()
    owner_member.userId = OWNER
    p.orgmember.find_first = AsyncMock(return_value=owner_member)
    p.orgbalance.find_unique = AsyncMock(return_value=MagicMock(balance=500))
    p.query_raw = AsyncMock(return_value=[{"balance": 400}])
    p.orgcredittransaction.create = AsyncMock()
    mocker.patch.object(org_credit, "prisma", p)
    return p


@pytest.fixture
def mock_user_credit(mocker):
    model = MagicMock()
    model.spend_credits = AsyncMock(return_value=1234)
    model.get_credits = AsyncMock(return_value=1234)
    model.grant_credits = AsyncMock(return_value=1234)
    mocker.patch.object(
        org_credit, "get_user_credit_model", AsyncMock(return_value=model)
    )
    return model


@pytest.mark.asyncio
async def test_personal_org_spend_debits_owner_wallet(mock_prisma, mock_user_credit):
    """The critical guard: a run in a personal org (i.e. every run after
    the silent migration) must debit the SAME user wallet it did before
    orgs existed — not the OrgBalance copy."""
    remaining = await spend_org_credits(
        ORG_PERSONAL, OWNER, 50, metadata={"reason": "block run"}
    )

    assert remaining == 1234
    mock_user_credit.spend_credits.assert_awaited_once()
    kwargs = mock_user_credit.spend_credits.await_args.kwargs
    assert kwargs["user_id"] == OWNER
    assert kwargs["cost"] == 50
    # OrgBalance untouched
    mock_prisma.query_raw.assert_not_called()
    mock_prisma.orgcredittransaction.create.assert_not_called()


@pytest.mark.asyncio
async def test_team_org_spend_debits_org_balance(mock_prisma, mock_user_credit):
    remaining = await spend_org_credits(ORG_TEAM, OWNER, 100)

    assert remaining == 400
    mock_user_credit.spend_credits.assert_not_called()
    mock_prisma.query_raw.assert_called_once()
    mock_prisma.orgcredittransaction.create.assert_called_once()


@pytest.mark.asyncio
async def test_personal_org_balance_reads_owner_wallet(mock_prisma, mock_user_credit):
    """Balance checks (dynamic-cost pre-flight guard) must see the wallet
    the spend will hit, or paid users get spurious InsufficientBalance."""
    balance = await get_org_credits(ORG_PERSONAL)

    assert balance == 1234
    mock_prisma.orgbalance.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_team_org_balance_reads_org_ledger(mock_prisma, mock_user_credit):
    balance = await get_org_credits(ORG_TEAM)

    assert balance == 500
    mock_user_credit.get_credits.assert_not_called()


@pytest.mark.asyncio
async def test_personal_org_top_up_credits_owner_wallet(mock_prisma, mock_user_credit):
    balance = await top_up_org_credits(
        ORG_PERSONAL, 200, user_id=OWNER, metadata={"reason": "promo"}
    )

    assert balance == 1234
    mock_user_credit.grant_credits.assert_awaited_once()
    kwargs = mock_user_credit.grant_credits.await_args.kwargs
    assert kwargs["user_id"] == OWNER
    assert kwargs["amount"] == 200
    mock_prisma.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_personal_org_missing_owner_falls_back_to_org_ledger(
    mock_prisma, mock_user_credit
):
    """A personal org with no owner row (data corruption) must not crash
    billing — fall back to the org ledger rather than dropping the charge."""
    mock_prisma.orgmember.find_first = AsyncMock(return_value=None)

    remaining = await spend_org_credits(ORG_PERSONAL, OWNER, 100)

    assert remaining == 400
    mock_user_credit.spend_credits.assert_not_called()
    mock_prisma.query_raw.assert_called_once()
