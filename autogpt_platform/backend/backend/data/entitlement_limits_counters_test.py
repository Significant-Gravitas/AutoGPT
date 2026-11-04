"""Each seat's Otto usage counts on its own in a team organization: apart from
the person's personal usage and from every other team organization they are in.
Personal-org (and no-org) usage keeps the counters it always had.

Only the stored state is stubbed (the spend counters, the usage state, the
organizations' plans); the recording, the checks, the usage display, the
resets and the budget gate run as in production."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import (
    budget_gate,
    org_cost_context,
    rate_limit,
    token_tracking,
    usage_activation,
)
from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import RateLimitExceeded, SubscriptionTier
from backend.copilot.trial_cost_context import (
    get_trial_cost_context,
    restore_cost_context,
)
from backend.data import entitlement, entitlement_limits
from backend.data.org_plan import OrgPlan, OrgPlanSnapshot
from backend.data.pro_activation import UsageActivationState
from backend.data.subscription_trial_integration_fixtures import pin_usage_clock

USER = "user-1"
PERSONAL, TEAM_A, TEAM_B = "personal-org", "team-a", "team-b"
DAILY, WEEKLY = 1_000_000, 5_000_000


class SpendCounters:
    """The Redis holding the spend counters: the reads, the INCRBY + EXPIRE
    transactions, the deletes and the floor-at-zero decrements the rate
    limiter issues. Anything else it would call is missing and fails."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        value = self.values.get(key)
        return None if value is None else str(value).encode()

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)

    async def eval(self, _script: str, _keys: int, key: str, delta: int) -> int:
        value = self.values.get(key, 0) - int(delta)
        if value < 0:
            self.values.pop(key, None)
            return 0
        self.values[key] = value
        return value

    def pipeline(self, transaction: bool) -> "_Transaction":
        assert transaction, "the spend counters are only written in a transaction"
        return _Transaction(self)


class _Transaction:
    def __init__(self, counters: SpendCounters) -> None:
        self._counters = counters
        self._increments: list[tuple[str, int]] = []

    def incrby(self, key: str, amount: int) -> None:
        self._increments.append((key, amount))

    def expire(self, key: str, seconds: int) -> None:
        assert seconds > 0

    async def execute(self) -> None:
        for key, amount in self._increments:
            self._counters.values[key] = self._counters.values.get(key, 0) + amount


def _team(org_id: str) -> OrgPlanSnapshot:
    plan = OrgPlan(
        tier=SubscriptionTier.PRO,
        seats=5,
        status="active",
        ends_at=None,
        note=None,
        updated_at=datetime.now(),
    )
    return OrgPlanSnapshot(
        org_id=org_id, exists=True, is_personal=False, is_live=True, plan=plan
    )


@pytest.fixture
def world(mocker, monkeypatch, set_platform_payment):
    now = pin_usage_clock(monkeypatch)
    counters = SpendCounters()
    mocker.patch.object(rate_limit, "get_redis_async", AsyncMock(return_value=counters))
    state = SimpleNamespace(tier=SubscriptionTier.PRO, trial_id=None, generation=None)
    usage = MagicMock(
        get_usage_activation_state=AsyncMock(
            side_effect=lambda user_id: UsageActivationState(
                user_id=user_id,
                tier=state.tier,
                trial_id=state.trial_id,
                generation=state.generation,
                ready=True,
            )
        )
    )
    mocker.patch.object(usage_activation, "pro_activation_db", return_value=usage)
    credits = MagicMock(record_subscription_trial_cost=AsyncMock())
    mocker.patch.object(rate_limit, "credit_db", return_value=credits)
    orgs = {
        PERSONAL: OrgPlanSnapshot(
            org_id=PERSONAL, exists=True, is_personal=True, is_live=True
        ),
        TEAM_A: _team(TEAM_A),
        TEAM_B: _team(TEAM_B),
    }
    mocker.patch.object(
        entitlement, "get_org_plan_snapshot", AsyncMock(side_effect=orgs.__getitem__)
    )
    mocker.patch.object(
        entitlement,
        "org_plan_db",
        return_value=MagicMock(query_active_membership=AsyncMock(return_value=True)),
    )
    set_platform_payment(True)
    # The flag service's limits: a base of DAILY and WEEKLY, PRO at 1x.
    mocker.patch.object(
        rate_limit,
        "_fetch_cost_limits_flag",
        AsyncMock(return_value={"daily": DAILY, "weekly": WEEKLY}),
    )
    mocker.patch.object(
        rate_limit, "_fetch_tier_multipliers_flag", AsyncMock(return_value={"PRO": 1})
    )
    mocker.patch.object(token_tracking, "_schedule_cost_log")
    return SimpleNamespace(now=now, counters=counters, state=state, credits=credits)


async def spend(usd: float, org_id: str | None) -> None:
    """An Otto call that cost *usd*, made in a chat in *org_id*."""
    session = ChatSession.new(USER, dry_run=False, organization_id=org_id)
    await token_tracking.persist_and_record_usage(
        session=session,
        user_id=USER,
        prompt_tokens=10,
        completion_tokens=10,
        cost_usd=usd,
    )


async def used(org_id: str | None) -> tuple[int, int]:
    """The daily and weekly usage the usage display reads in *org_id*."""
    daily, weekly, tier = await entitlement_limits.get_global_rate_limits(
        USER, DAILY, WEEKLY, org_id
    )
    status = await entitlement_limits.get_usage_status(
        USER, daily, weekly, tier=tier, org_id=org_id
    )
    return status.daily.used, status.weekly.used


async def check(org_id: str | None) -> None:
    daily, weekly, tier = await entitlement_limits.get_global_rate_limits(
        USER, DAILY, WEEKLY, org_id
    )
    await entitlement_limits.check_rate_limit(
        USER, daily, weekly, expected_tier=tier, org_id=org_id
    )


def personal_keys(now: datetime, generation: str | None = None) -> set[str]:
    """The keys personal usage has always had."""
    owner = f"{USER}:{generation}" if generation else USER
    year, week, _ = now.isocalendar()
    return {
        f"copilot:cost:daily:{owner}:{now.strftime('%Y-%m-%d')}",
        f"copilot:cost:weekly:{owner}:{year}-W{week:02d}",
    }


@pytest.mark.asyncio
async def test_team_org_usage_leaves_personal_usage_alone(world):
    await spend(0.25, TEAM_A)
    assert await used(TEAM_A) == (250_000, 250_000)
    assert await used(PERSONAL) == (0, 0)
    assert await used(None) == (0, 0)


@pytest.mark.asyncio
async def test_personal_usage_leaves_team_org_usage_alone(world):
    await spend(0.25, TEAM_A)
    await spend(0.1, PERSONAL)
    await spend(0.05, None)
    assert await used(PERSONAL) == (150_000, 150_000)
    assert await used(None) == (150_000, 150_000)
    assert await used(TEAM_A) == (250_000, 250_000)


@pytest.mark.asyncio
async def test_two_team_orgs_count_apart(world):
    await spend(0.25, TEAM_A)
    await spend(0.5, TEAM_B)
    assert await used(TEAM_A) == (250_000, 250_000)
    assert await used(TEAM_B) == (500_000, 500_000)


@pytest.mark.asyncio
@pytest.mark.parametrize("org_id", [PERSONAL, None])
@pytest.mark.parametrize("generation", [None, "paid-1"])
async def test_personal_usage_keeps_the_keys_it_always_had(world, org_id, generation):
    world.state.generation = generation
    await spend(0.1, org_id)
    assert set(world.counters.values) == personal_keys(world.now, generation)


@pytest.mark.asyncio
async def test_team_org_usage_is_keyed_by_person_and_org(world):
    world.state.generation = "paid-1"
    await spend(0.1, TEAM_A)
    year, week, _ = world.now.isocalendar()
    assert set(world.counters.values) == {
        f"copilot:cost:daily:{USER}:org:{TEAM_A}:{world.now.strftime('%Y-%m-%d')}",
        f"copilot:cost:weekly:{USER}:org:{TEAM_A}:{year}-W{week:02d}",
    }


@pytest.mark.asyncio
async def test_a_spent_team_allowance_refuses_only_in_that_org(world):
    await spend(1.0, TEAM_A)
    with pytest.raises(RateLimitExceeded):
        await check(TEAM_A)
    await check(TEAM_B)
    await check(PERSONAL)
    await check(None)


@pytest.mark.asyncio
async def test_a_spent_personal_allowance_refuses_only_personally(world):
    await spend(1.0, PERSONAL)
    with pytest.raises(RateLimitExceeded):
        await check(PERSONAL)
    with pytest.raises(RateLimitExceeded):
        await check(None)
    await check(TEAM_A)


@pytest.mark.asyncio
async def test_the_budget_gate_pauses_only_in_the_spent_org(world):
    await spend(1.0, TEAM_A)
    pause = await budget_gate.refusal_for(USER, TEAM_A)
    assert pause is not None and pause.reason == "daily"
    assert await budget_gate.refusal_for(USER, PERSONAL) is None
    assert await budget_gate.refusal_for(USER, TEAM_B) is None


@pytest.mark.asyncio
async def test_the_remaining_budget_is_the_orgs_own(world):
    await spend(0.75, TEAM_A)
    remaining = await entitlement_limits.get_remaining_usd_budget(
        USER, DAILY, WEEKLY, floor_usd=0.0, org_id=TEAM_A
    )
    assert remaining == pytest.approx(0.25)
    personal = await entitlement_limits.get_remaining_usd_budget(
        USER, DAILY, WEEKLY, floor_usd=0.0, org_id=PERSONAL
    )
    assert personal == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_an_admin_reset_clears_only_that_orgs_counters(world):
    await spend(0.25, TEAM_A)
    await spend(0.5, TEAM_B)
    await spend(0.1, PERSONAL)
    await entitlement_limits.reset_user_usage(USER, reset_weekly=True, org_id=TEAM_A)
    assert await used(TEAM_A) == (0, 0)
    assert await used(TEAM_B) == (500_000, 500_000)
    assert await used(PERSONAL) == (100_000, 100_000)
    await entitlement_limits.reset_user_usage(USER, org_id=None)
    assert await used(PERSONAL) == (0, 100_000)
    assert await used(TEAM_B) == (500_000, 500_000)


@pytest.mark.asyncio
async def test_a_paid_daily_reset_clears_only_that_orgs_counters(world):
    await spend(1.0, TEAM_A)
    await spend(1.0, PERSONAL)
    assert await entitlement_limits.reset_daily_usage(USER, DAILY, org_id=PERSONAL)
    assert await used(PERSONAL) == (0, 0)
    assert await used(TEAM_A) == (1_000_000, 1_000_000)


@pytest.mark.asyncio
async def test_work_started_in_a_team_org_counts_there_wherever_it_runs(world):
    """A turn, a held call or a background job records under the attribution
    captured where it started, not the chat it happens to record from."""
    attribution = await org_cost_context.capture_org_cost_context(USER, TEAM_A)
    with restore_cost_context(USER, attribution):
        await spend(0.25, None)
    assert await used(TEAM_A) == (250_000, 250_000)
    assert await used(None) == (0, 0)


@pytest.mark.asyncio
async def test_background_work_counts_in_the_org_it_is_given(world):
    @org_cost_context.attributed_usage
    async def simulate(*, user_id: str, org_id: str | None) -> None:
        await spend(0.25, None)

    @org_cost_context.attributed_usage
    async def dream(user_id: str, *, organization_id: str | None = None) -> None:
        await spend(0.5, None)

    await simulate(user_id=USER, org_id=TEAM_A)
    await dream(USER, organization_id=TEAM_B)
    await dream(USER)
    assert await used(TEAM_A) == (250_000, 250_000)
    assert await used(TEAM_B) == (500_000, 500_000)
    assert await used(None) == (500_000, 500_000)
    assert get_trial_cost_context(USER) is None


@pytest.mark.asyncio
async def test_team_org_work_never_spends_a_personal_trial(world):
    world.state.tier, world.state.trial_id = SubscriptionTier.TRIAL, "trial-1"
    await spend(0.25, TEAM_A)
    world.credits.record_subscription_trial_cost.assert_not_awaited()
    await spend(0.1, PERSONAL)
    world.credits.record_subscription_trial_cost.assert_awaited_once_with(
        USER, 100_000, trial_id="trial-1"
    )


@pytest.mark.asyncio
async def test_a_team_org_attribution_names_no_personal_trial(world):
    world.state.tier, world.state.trial_id = SubscriptionTier.TRIAL, "trial-1"
    team = await org_cost_context.capture_org_cost_context(USER, TEAM_A)
    assert (team.team_org_id, team.trial_id, team.generation) == (TEAM_A, None, None)
    own = await org_cost_context.capture_org_cost_context(USER, PERSONAL)
    assert (own.team_org_id, own.trial_id) == (None, "trial-1")
