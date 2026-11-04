"""Members plus pending invitations may not exceed a team plan's seats."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest

from backend.api.features.orgs import seats
from backend.copilot.rate_limit import SubscriptionTier
from backend.data.org_plan import OrgPlan, OrgPlanSnapshot


def _snapshot(plan_seats: int | None) -> OrgPlanSnapshot:
    plan = (
        OrgPlan(
            tier=SubscriptionTier.BUSINESS,
            seats=plan_seats,
            status="active",
            ends_at=None,
            note=None,
            updated_at=datetime.now(timezone.utc),
        )
        if plan_seats is not None
        else None
    )
    return OrgPlanSnapshot(org_id="org", exists=True, is_live=True, plan=plan)


@pytest.fixture
def org(mocker, set_platform_payment):
    """A team org with one member (its owner) and payments on; tests set the
    plan's seats and the pending invitations."""
    client = MagicMock()
    client.orgmember.count = AsyncMock(return_value=1)
    client.orginvitation.count = AsyncMock(return_value=0)
    lock = mocker.patch.object(seats, "lock_org_membership", AsyncMock())
    set_platform_payment(True)
    plan = mocker.patch.object(
        seats, "read_org_plan_snapshot", AsyncMock(return_value=_snapshot(3))
    )
    return client, plan, lock


@pytest.mark.asyncio
async def test_invitations_fill_the_seats_beside_the_owner(org):
    client, _, lock = org
    client.orginvitation.count.return_value = 1
    await seats.assert_free_seat(client, "org", "owner", accepting=False)
    lock.assert_awaited_once_with(client, "org", "owner")

    client.orginvitation.count.return_value = 2
    with pytest.raises(fastapi.HTTPException) as refused:
        await seats.assert_free_seat(client, "org", "owner", accepting=False)
    assert refused.value.status_code == 409
    assert "All 3 seats" in refused.value.detail


@pytest.mark.asyncio
async def test_accepting_moves_an_invitations_seat_to_the_new_member(org):
    client, _, _ = org
    client.orginvitation.count.return_value = 2
    await seats.assert_free_seat(client, "org", "invitee", accepting=True)


@pytest.mark.asyncio
async def test_accepting_is_refused_when_the_seats_were_cut_below_the_invitations(org):
    client, plan, _ = org
    client.orginvitation.count.return_value = 2
    plan.return_value = _snapshot(2)
    with pytest.raises(fastapi.HTTPException):
        await seats.assert_free_seat(client, "org", "invitee", accepting=True)


@pytest.mark.asyncio
async def test_without_a_plan_nobody_can_be_invited(org):
    client, plan, _ = org
    plan.return_value = _snapshot(None)
    with pytest.raises(fastapi.HTTPException) as refused:
        await seats.assert_free_seat(client, "org", "owner", accepting=False)
    assert refused.value.detail == seats.NO_PLAN_REFUSAL


@pytest.mark.asyncio
async def test_with_payments_off_there_are_no_seats_to_count(org, set_platform_payment):
    client, plan, lock = org
    set_platform_payment(False)
    await seats.assert_free_seat(client, "org", "owner", accepting=False)
    plan.assert_not_awaited()
    lock.assert_not_awaited()
