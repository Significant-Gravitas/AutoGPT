"""Real Postgres: the clock only moves forwards, and the usage state carries it."""

import asyncio

import pytest
from prisma.models import UsageBillingClock

from backend.data import usage_clock_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.usage_clock import save_billing_clock

billing_user = fixtures.billing_user
pytestmark = fixtures.pytestmark

DAY = 86_400
START = 1_793_824_256
NOVEMBER = {
    "id": "sub_new",
    "customer": "cus_x",
    "current_period_start": START,
    "current_period_end": START + 30 * DAY,
    "billing_cycle_anchor": START,
}
OCTOBER = {
    **NOVEMBER,
    "current_period_start": START - 31 * DAY,
    "current_period_end": START,
}


async def _stored(user_id: str) -> tuple[str, int, int]:
    row = await UsageBillingClock.prisma().find_unique_or_raise(
        where={"userId": user_id}
    )
    return row.stripeSubscriptionId, row.periodStart, row.periodEnd


@pytest.mark.asyncio
async def test_a_period_is_copied_and_read_back_with_the_usage_state(billing_user):
    assert await save_billing_clock(billing_user, NOVEMBER) is True

    state = await get_usage_activation_state(billing_user)

    assert (state.period_start, state.period_end) == (START, START + 30 * DAY)
    assert state.billing_anchor == START
    assert state.has_stripe_customer is True
    assert state.created_at is not None and state.created_at > 1_700_000_000
    assert (state.extra_remaining, state.extra_total, state.extra_count) == (0, 0, 0)
    assert state.extra_expires_at is None


@pytest.mark.asyncio
async def test_a_user_without_a_clock_still_has_a_state(billing_user):
    state = await get_usage_activation_state(billing_user)

    assert state.period_start is None and state.period_end is None
    assert state.tier.value == "PRO" and state.ready


@pytest.mark.asyncio
async def test_an_older_period_never_replaces_a_newer_one(billing_user):
    """A late or duplicate delivery, or an old plan revived beside its
    replacement, carries a period that started earlier."""
    await save_billing_clock(billing_user, NOVEMBER)

    assert await save_billing_clock(billing_user, {**OCTOBER, "id": "sub_old"}) is False

    assert await _stored(billing_user) == ("sub_new", START, START + 30 * DAY)


@pytest.mark.asyncio
async def test_a_renewal_an_upgrade_and_a_new_plan_move_the_clock_forwards(
    billing_user,
):
    await save_billing_clock(billing_user, OCTOBER)
    assert await save_billing_clock(billing_user, NOVEMBER) is True  # a renewal
    upgraded = {
        **NOVEMBER,
        "current_period_start": START + DAY,
        "current_period_end": START + 31 * DAY,
    }
    assert await save_billing_clock(billing_user, upgraded) is True  # reset to now
    replaced = {**upgraded, "id": "sub_other", "current_period_start": START + 2 * DAY}
    assert await save_billing_clock(billing_user, replaced) is True  # a new plan

    assert await _stored(billing_user) == (
        "sub_other",
        START + 2 * DAY,
        START + 31 * DAY,
    )


@pytest.mark.asyncio
async def test_the_same_period_again_is_harmless_and_takes_a_changed_end(billing_user):
    await save_billing_clock(billing_user, NOVEMBER)
    shortened = {**NOVEMBER, "current_period_end": START + 20 * DAY}

    assert await save_billing_clock(billing_user, NOVEMBER) is True
    assert await save_billing_clock(billing_user, shortened) is True

    assert await _stored(billing_user) == ("sub_new", START, START + 20 * DAY)


@pytest.mark.asyncio
async def test_writers_finishing_in_either_order_end_on_the_newer_period(billing_user):
    await asyncio.gather(
        save_billing_clock(billing_user, NOVEMBER),
        save_billing_clock(billing_user, OCTOBER),
        save_billing_clock(billing_user, NOVEMBER),
        save_billing_clock(billing_user, OCTOBER),
    )

    assert (await _stored(billing_user))[1] == START


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        {"current_period_start": START * 1000, "current_period_end": START * 1000 + 5},
        {"current_period_end": START},
        {"current_period_start": None},
        {"current_period_start": "1793824256"},
    ],
)
async def test_a_period_that_is_not_sane_seconds_changes_nothing(billing_user, broken):
    await save_billing_clock(billing_user, NOVEMBER)

    assert await save_billing_clock(billing_user, {**NOVEMBER, **broken}) is False

    assert await _stored(billing_user) == ("sub_new", START, START + 30 * DAY)
