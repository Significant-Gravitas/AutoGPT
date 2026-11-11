"""Real Postgres: an extra week is charged and created together or not at
all, bought once per key, and drawn on oldest-expiry first."""

import asyncio

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import CreditTransaction, UsageExtraAllowance

from backend.data import usage_clock_integration_fixtures as fixtures
from backend.data import usage_extra
from backend.data.credit import UserCredit
from backend.data.pro_activation import get_usage_activation_state
from backend.data.usage_extra import (
    EXTRA_WEEK_SECONDS,
    buy_extra_week,
    consume_extra_allowance,
    credit_transaction_key,
    list_extra_weeks,
)
from backend.util.exceptions import InsufficientBalanceError

billing_user = fixtures.billing_user
pytestmark = fixtures.pytestmark

NOW = 1_793_824_256
PRICE, AMOUNT = 1_250, 8_000_000


async def _buy(user_id: str, key: str = "key-1", now: int = NOW, amount: int = AMOUNT):
    return await buy_extra_week(
        user_id,
        key,
        tier=SubscriptionTier.PRO,
        amount=amount,
        price_credits=PRICE,
        now=now,
    )


async def _balance(user_id: str) -> int:
    return await UserCredit().get_credits(user_id)


async def _charges(user_id: str) -> list[CreditTransaction]:
    return await CreditTransaction.prisma().find_many(
        where={"userId": user_id, "transactionKey": {"startsWith": "EXTRA-WEEK-"}}
    )


@pytest.mark.asyncio
async def test_buying_takes_the_credits_and_gives_the_week_for_thirty_days(
    billing_user,
):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    purchase = await _buy(billing_user)

    assert purchase.charged and purchase.balance == 3_750
    assert await _balance(billing_user) == 3_750
    week = purchase.extra_week
    assert (week.amount, week.used, week.price_credits) == (AMOUNT, 0, PRICE)
    assert week.expires_at - week.purchased_at == EXTRA_WEEK_SECONDS == 2_592_000
    charge = (await _charges(billing_user))[0]
    assert charge.amount == -PRICE
    assert charge.transactionKey == credit_transaction_key("key-1")
    state = await get_usage_activation_state(billing_user)
    assert (state.extra_remaining, state.extra_total, state.extra_count) == (
        AMOUNT,
        AMOUNT,
        1,
    )


@pytest.mark.asyncio
async def test_too_few_credits_buys_nothing_and_charges_nothing(billing_user):
    await UserCredit().grant_credits(billing_user, PRICE - 1, "test balance")

    with pytest.raises(InsufficientBalanceError):
        await _buy(billing_user)

    assert await _balance(billing_user) == PRICE - 1
    assert await _charges(billing_user) == []
    assert await list_extra_weeks(billing_user, NOW) == []


@pytest.mark.asyncio
async def test_the_same_key_again_returns_the_purchase_and_charges_once(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    first = await _buy(billing_user)

    again = await _buy(billing_user)

    assert not again.charged
    assert again.extra_week == first.extra_week
    assert again.balance == await _balance(billing_user) == 3_750
    assert len(await _charges(billing_user)) == 1


@pytest.mark.asyncio
async def test_the_same_key_twice_at_once_charges_once(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    purchases = await asyncio.gather(*(_buy(billing_user) for _ in range(4)))

    assert sum(purchase.charged for purchase in purchases) == 1
    assert len({purchase.extra_week.id for purchase in purchases}) == 1
    assert await _balance(billing_user) == 3_750
    assert len(await list_extra_weeks(billing_user, NOW)) == 1


@pytest.mark.asyncio
async def test_a_failure_after_the_charge_rolls_the_charge_back(
    billing_user, monkeypatch: pytest.MonkeyPatch
):
    """Charged without the allowance must be impossible: both are one
    transaction."""
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    monkeypatch.setattr(usage_extra, "EXTRA_WEEK_SECONDS", None)  # the insert fails

    with pytest.raises(TypeError):
        await _buy(billing_user)

    assert await _balance(billing_user) == 5_000
    assert await _charges(billing_user) == []
    assert await UsageExtraAllowance.prisma().count(where={"userId": billing_user}) == 0


@pytest.mark.asyncio
async def test_extra_weeks_stack_and_the_one_expiring_first_is_used_first(
    billing_user,
):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    later = (await _buy(billing_user, "later", NOW + 500)).extra_week
    sooner = (await _buy(billing_user, "sooner", NOW)).extra_week

    assert (
        await consume_extra_allowance(billing_user, 3_000_000, NOW + 600) == 3_000_000
    )
    assert (
        await consume_extra_allowance(billing_user, 7_000_000, NOW + 700) == 7_000_000
    )

    weeks = {week.id: week for week in await list_extra_weeks(billing_user, NOW + 800)}
    assert weeks[sooner.id].used == AMOUNT  # used up first
    assert weeks[later.id].used == 2_000_000
    state = await get_usage_activation_state(billing_user)
    assert (state.extra_remaining, state.extra_count) == (6_000_000, 1)


@pytest.mark.asyncio
async def test_no_more_is_taken_than_is_left(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    await _buy(billing_user)

    assert await consume_extra_allowance(billing_user, 9_000_000, NOW + 5) == AMOUNT
    assert await consume_extra_allowance(billing_user, 1, NOW + 6) == 0
    assert await consume_extra_allowance(billing_user, 0, NOW + 6) == 0


@pytest.mark.asyncio
async def test_costs_recorded_at_once_never_spend_the_same_remainder(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    await _buy(billing_user)

    taken = await asyncio.gather(
        *(consume_extra_allowance(billing_user, 3_000_000, NOW + 5) for _ in range(5))
    )

    assert sum(taken) == AMOUNT
    assert (await list_extra_weeks(billing_user, NOW + 5))[0].used == AMOUNT


@pytest.mark.asyncio
async def test_an_expired_week_is_neither_used_nor_counted(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    await _buy(billing_user)
    expiry = NOW + EXTRA_WEEK_SECONDS

    assert await consume_extra_allowance(billing_user, 1_000, expiry) == 0
    assert await consume_extra_allowance(billing_user, 1_000, expiry - 1) == 1_000
    assert await list_extra_weeks(billing_user, expiry) == []


@pytest.mark.asyncio
async def test_an_extra_week_goes_with_its_account(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    await _buy(billing_user)

    await fixtures.remove_user(billing_user)

    assert await UsageExtraAllowance.prisma().count(where={"userId": billing_user}) == 0
