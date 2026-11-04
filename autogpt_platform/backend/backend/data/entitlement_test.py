"""The one entitlement check: the user's own plan in a personal organization,
the team organization's plan (and an active membership) in a team one."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import pytest
from autogpt_libs.auth.models import RequestContext

from backend.copilot.rate_limit import SubscriptionTier
from backend.data import entitlement
from backend.data.org_plan import OrgPlan, OrgPlanSnapshot
from backend.util.exceptions import UserPaywalledError

NOW = datetime.now(timezone.utc)


def _plan(**changes) -> OrgPlan:
    fields = {
        "tier": SubscriptionTier.BUSINESS,
        "seats": 3,
        "status": "active",
        "ends_at": None,
        "note": None,
        "updated_at": NOW,
    }
    return OrgPlan(**(fields | changes))


def _team(plan: OrgPlan | None = None, **changes) -> OrgPlanSnapshot:
    fields = {"org_id": "team-org", "exists": True, "is_personal": False}
    return OrgPlanSnapshot(**(fields | {"is_live": True, "plan": plan} | changes))


@pytest.fixture
def world(mocker):
    """The user's own plan, the org, membership and the payments setting."""
    state = SimpleNamespace(
        own_paywalled=False,
        own_tier=SubscriptionTier.PRO,
        snapshot=_team(_plan()),
        member=True,
        payments=True,
    )
    mocker.patch.object(
        entitlement.rate_limit,
        "is_user_paywalled",
        AsyncMock(side_effect=lambda _u: state.own_paywalled),
    )
    mocker.patch.object(
        entitlement.rate_limit,
        "get_user_tier",
        AsyncMock(side_effect=lambda _u: state.own_tier),
    )
    mocker.patch.object(
        entitlement,
        "get_org_plan_snapshot",
        AsyncMock(side_effect=lambda _o: state.snapshot),
    )
    membership = SimpleNamespace(
        query_active_membership=AsyncMock(side_effect=lambda _o, _u: state.member)
    )
    mocker.patch.object(entitlement, "org_plan_db", return_value=membership)
    mocker.patch.object(
        entitlement,
        "is_platform_payment_enabled",
        side_effect=lambda: state.payments,
    )
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("org_id", [None, "personal-org"])
async def test_no_org_or_a_personal_org_uses_the_users_own_plan(world, org_id):
    world.snapshot = OrgPlanSnapshot(
        org_id="personal-org", exists=True, is_personal=True, is_live=True
    )
    world.own_paywalled, world.own_tier = True, SubscriptionTier.NO_TIER

    result = await entitlement.get_entitlement("u", org_id)

    assert result.scope == "personal"
    assert result.paywall == "no_plan"
    assert await entitlement.get_user_tier("u", org_id) == SubscriptionTier.NO_TIER


@pytest.mark.asyncio
async def test_a_team_plan_gives_its_tier_to_a_member_without_one(world):
    world.own_paywalled, world.own_tier = True, SubscriptionTier.NO_TIER

    result = await entitlement.get_entitlement("u", "team-org")

    assert (result.scope, result.paywall) == ("organization", None)
    assert result.tier == SubscriptionTier.BUSINESS
    assert await entitlement.get_user_tier("u", "team-org") == SubscriptionTier.BUSINESS


@pytest.mark.asyncio
async def test_personal_pro_does_not_count_in_a_team_org_without_a_plan(world):
    world.snapshot = _team(None)

    reason = await entitlement.paywall_reason("u", "team-org")

    assert reason == "org_plan_required"
    assert await entitlement.get_user_tier("u", "team-org") == SubscriptionTier.NO_TIER
    assert "organization needs an active plan" in entitlement.paywall_message(reason)
    assert "Upgrade" not in entitlement.paywall_message(reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan",
    [
        _plan(ends_at=NOW - timedelta(minutes=1)),
        _plan(status="canceled"),
    ],
    ids=["past-its-end-date", "not-active"],
)
async def test_a_plan_that_doesnt_count_paywalls_the_org(world, plan):
    world.snapshot = _team(plan)
    assert await entitlement.is_user_paywalled("u", "team-org")


@pytest.mark.asyncio
async def test_a_plan_with_an_end_date_ahead_counts(world):
    world.snapshot = _team(_plan(ends_at=NOW + timedelta(days=30)))
    assert not await entitlement.is_user_paywalled("u", "team-org")


@pytest.mark.asyncio
async def test_someone_who_isnt_a_member_is_refused_even_on_a_plan(world):
    world.member = False
    assert await entitlement.paywall_reason("u", "team-org") == "not_member"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        _team(_plan(), is_live=False),
        OrgPlanSnapshot(org_id="team-org", exists=False),
    ],
    ids=["archived-or-deleted", "missing"],
)
async def test_an_org_that_is_gone_refuses(world, snapshot):
    world.snapshot = snapshot
    assert await entitlement.paywall_reason("u", "team-org") == "org_inactive"


@pytest.mark.asyncio
async def test_with_payments_off_a_member_is_not_paywalled_without_a_plan(world):
    world.snapshot, world.payments = _team(None), False
    assert await entitlement.paywall_reason("u", "team-org") is None


@pytest.mark.asyncio
async def test_background_work_raises_the_orgs_refusal(world):
    world.snapshot = _team(None)
    with pytest.raises(UserPaywalledError, match="to run workflows"):
        await entitlement.require_entitlement("u", "team-org", action="run workflows")


@pytest.mark.asyncio
async def test_the_http_gate_answers_503_when_the_plan_cant_be_read(world, mocker):
    mocker.patch.object(
        entitlement, "get_org_plan_snapshot", AsyncMock(side_effect=RuntimeError)
    )
    with pytest.raises(fastapi.HTTPException) as error:
        await entitlement.enforce_entitlement("u", "team-org")
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "30"}


@pytest.mark.asyncio
async def test_the_route_dependency_reads_the_verified_context(world):
    world.snapshot = _team(None)
    ctx = RequestContext(
        user_id="u",
        org_id="team-org",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    with pytest.raises(UserPaywalledError, match="organization needs"):
        await entitlement.enforce_payment_paywall(ctx)
