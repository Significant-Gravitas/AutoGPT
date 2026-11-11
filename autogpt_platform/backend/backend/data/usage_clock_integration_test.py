"""Real Postgres: the clock only moves forwards, and the usage state carries it."""

import asyncio
import time

import pytest
from prisma.models import UsageBillingClock

from backend.data import usage_clock_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.usage_clock import save_billing_clock

billing_user = fixtures.billing_user
pytestmark = fixtures.pytestmark

DAY = 86_400
# The stored period is compared with the database's clock, so the periods
# here are placed around the real moment: this one began six days ago.
START = int(time.time()) - 6 * DAY
NOVEMBER = {
    "id": "sub_new",
    "customer": "cus_x",
    "current_period_start": START,
    "current_period_end": START + 30 * DAY,
    "billing_cycle_anchor": START,
    "created": START - 90 * DAY,
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
    replaced = {
        **upgraded,
        "id": "sub_other",
        "current_period_start": START + 2 * DAY,
        "created": START + 2 * DAY,
    }
    assert await save_billing_clock(billing_user, replaced) is True  # a new plan

    assert await _stored(billing_user) == (
        "sub_other",
        START + 2 * DAY,
        START + 31 * DAY,
    )


def _plan(plan_id: str, created: int, start: int, days: int = 30) -> dict:
    return {
        "id": plan_id,
        "customer": "cus_x",
        "created": created,
        "current_period_start": start,
        "current_period_end": start + days * DAY,
        "billing_cycle_anchor": created,
    }


@pytest.mark.asyncio
async def test_an_older_plan_left_running_never_takes_the_clock_from_the_newer(
    billing_user,
):
    """Two live plans on one customer renew a few days apart. If each
    renewal took the clock, every one would start a fresh first window."""
    now = int(time.time())
    newer = _plan("sub_newer", created=now - 40 * DAY, start=now - 10 * DAY)
    older_renewed = _plan("sub_older", created=now - 300 * DAY, start=now - 2 * DAY)
    await save_billing_clock(billing_user, newer)

    assert await save_billing_clock(billing_user, older_renewed) is False

    assert (await _stored(billing_user))[0] == "sub_newer"


@pytest.mark.asyncio
async def test_when_the_newer_plan_ends_the_clock_passes_to_the_one_that_remains(
    billing_user,
):
    """The newer plan was cancelled; its last period is over. The older plan,
    whose current period started before that, is the live one: the customer
    must not be held on a plan that no longer exists."""
    now = int(time.time())
    ended = _plan("sub_newer", created=now - 70 * DAY, start=now - 31 * DAY)
    remaining = _plan(
        "sub_older", created=now - 300 * DAY, start=now - 200 * DAY, days=365
    )
    await save_billing_clock(billing_user, ended)

    assert await save_billing_clock(billing_user, remaining) is True

    stored = await _stored(billing_user)
    assert stored == ("sub_older", now - 200 * DAY, now + 165 * DAY)


@pytest.mark.asyncio
async def test_a_newer_plan_takes_the_clock_while_the_old_period_still_runs(
    billing_user,
):
    now = int(time.time())
    old = _plan("sub_old", created=now - 300 * DAY, start=now - 10 * DAY)
    new = _plan("sub_new", created=now - 60, start=now - 60)
    await save_billing_clock(billing_user, old)

    assert await save_billing_clock(billing_user, new) is True

    assert (await _stored(billing_user))[0] == "sub_new"


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
