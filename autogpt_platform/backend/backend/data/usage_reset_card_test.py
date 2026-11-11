"""The card path's rarer turns, with the database and Stripe stood in for.
Its main flows run against Postgres in usage_reset_integration_test.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier
from prisma.errors import UniqueViolationError
from redis.exceptions import RedisError

from backend.data import usage_reset, usage_reset_card
from backend.data.usage_reset import ResetPurchase, announce_reset
from backend.data.usage_reset_card import fulfill_card_reset, start_card_reset

NOW = 1_793_824_256
TAKEN = UniqueViolationError({"user_facing_error": {"meta": {"target": ["x"]}}})


def _row(status: str = "pending") -> MagicMock:
    return MagicMock(
        id="reset-1",
        userId="user-1",
        purchaseKey="key-card-1",
        tier="MAX",
        priceCredits=8_000,
        paidWith="card",
        status=status,
        stripeCheckoutSessionId="cs_test_1",
        stripePaymentIntentId=None,
        weekStart=None,
        weekEnd=None,
        sequence=None,
        grantedAt=None,
    )


def _paid_session() -> dict:
    return {
        "id": "cs_test_1",
        "mode": "payment",
        "payment_status": "paid",
        "amount_subtotal": 8_000,
        "payment_intent": "pi_test_1",
        "metadata": {
            "usage_reset": "1",
            "user_id": "user-1",
            "purchase_key": "key-card-1",
        },
    }


@pytest.fixture
def table(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    table = MagicMock()
    table.find_first = AsyncMock(return_value=_row())
    table.find_unique_or_raise = AsyncMock(return_value=_row("granted"))
    table.count = AsyncMock(return_value=0)
    table.update_many = AsyncMock(return_value=1)
    table.create = AsyncMock()
    monkeypatch.setattr(
        usage_reset_card.UsageReset, "prisma", MagicMock(return_value=table)
    )
    monkeypatch.setattr(
        usage_reset_card, "stripe_call", AsyncMock(return_value=_paid_session())
    )
    monkeypatch.setattr(usage_reset_card, "announce_reset", AsyncMock())
    return table


@pytest.mark.asyncio
async def test_a_paid_reset_takes_the_next_number_when_its_own_was_just_taken(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """Two paid resets confirmed at the same moment: both were paid for, so
    both are granted, one after the other."""
    monkeypatch.setattr(usage_reset_card, "get_usage_activation_state", AsyncMock())
    windows = MagicMock(week_start=NOW - 86_400, week_end=NOW + 86_400)
    monkeypatch.setattr(
        usage_reset_card, "clock_from_state", lambda *_: MagicMock(windows=windows)
    )
    table.count.side_effect = [0, 1]
    table.update_many.side_effect = [TAKEN, 1]

    grant = await fulfill_card_reset("cs_test_1", NOW)

    assert grant is not None and grant.granted_now
    first, second = table.update_many.await_args_list
    assert first.kwargs["data"]["sequence"] == 1
    assert second.kwargs["data"]["sequence"] == 2
    # Only ever a row that is still pending.
    assert second.kwargs["where"] == {"id": "reset-1", "status": "pending"}


@pytest.mark.asyncio
async def test_a_paid_reset_that_cannot_be_numbered_is_an_error_stripe_retries(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(usage_reset_card, "get_usage_activation_state", AsyncMock())
    monkeypatch.setattr(usage_reset_card, "clock_from_state", lambda *_: MagicMock())
    table.update_many.side_effect = TAKEN

    with pytest.raises(RuntimeError):
        await fulfill_card_reset("cs_test_1", NOW)


@pytest.mark.asyncio
async def test_a_paid_reset_with_no_window_to_go_in_is_an_error_not_a_loss(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """Paid for, but the account has no clock to place it on: raising makes
    Stripe deliver the event again and leaves the purchase pending and
    visible, instead of answering 200 and dropping it."""
    monkeypatch.setattr(usage_reset_card, "get_usage_activation_state", AsyncMock())
    monkeypatch.setattr(usage_reset_card, "clock_from_state", lambda *_: None)

    with pytest.raises(RuntimeError):
        await fulfill_card_reset("cs_test_1", NOW)

    table.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_requests_opening_the_same_purchase_share_one_checkout(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """Both found no purchase and both opened a session; the second to store
    its own is given the first's instead, so only one can ever be paid."""
    first = ResetPurchase(
        id="reset-1",
        user_id="user-1",
        tier="MAX",
        price_credits=8_000,
        paid_with="card",
        status="pending",
        checkout_session_id="cs_test_first",
    )
    monkeypatch.setattr(
        usage_reset_card, "find_reset_purchase", AsyncMock(side_effect=[None, first])
    )
    mine = MagicMock(id="cs_test_mine", url="https://checkout.stripe.test/mine")
    theirs = {"id": "cs_test_first", "url": "https://checkout.stripe.test/first"}
    monkeypatch.setattr(
        usage_reset_card, "stripe_call", AsyncMock(side_effect=[mine, theirs])
    )
    monkeypatch.setattr(
        usage_reset_card, "get_stripe_customer_id", AsyncMock(return_value="cus_1")
    )
    table.create.side_effect = TAKEN

    checkout = await start_card_reset(
        "user-1", "key-card-1", tier=SubscriptionTier.MAX, price_credits=8_000, now=NOW
    )

    assert checkout.url == "https://checkout.stripe.test/first"
    assert checkout.purchase.id == "reset-1"


@pytest.mark.asyncio
async def test_a_key_credits_paid_for_while_its_checkout_was_opening_is_not_sold_twice(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """Credits and the card raced for one key and credits won: the session
    opened here is never shown, and the answer is the granted purchase."""
    paid = ResetPurchase(
        id="reset-1",
        user_id="user-1",
        tier="MAX",
        price_credits=8_000,
        paid_with="credits",
        status="granted",
        week_start=NOW - 86_400,
        week_end=NOW + 86_400,
        sequence=1,
        granted_at=NOW,
    )
    monkeypatch.setattr(
        usage_reset_card, "find_reset_purchase", AsyncMock(side_effect=[None, paid])
    )
    monkeypatch.setattr(
        usage_reset_card,
        "stripe_call",
        AsyncMock(return_value=MagicMock(id="cs_test_mine", url="https://x.test")),
    )
    monkeypatch.setattr(
        usage_reset_card, "get_stripe_customer_id", AsyncMock(return_value="cus_1")
    )
    table.create.side_effect = TAKEN

    checkout = await start_card_reset(
        "user-1", "key-card-1", tier=SubscriptionTier.MAX, price_credits=8_000, now=NOW
    )

    assert checkout.url is None and checkout.purchase.status == "granted"


@pytest.mark.asyncio
async def test_a_checkout_needs_a_price_and_a_url(
    table: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        usage_reset_card, "find_reset_purchase", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        usage_reset_card, "get_stripe_customer_id", AsyncMock(return_value="cus_1")
    )
    monkeypatch.setattr(
        usage_reset_card,
        "stripe_call",
        AsyncMock(return_value=MagicMock(id="cs_test_2", url=None)),
    )

    with pytest.raises(ValueError):
        await start_card_reset(
            "user-1", "key-card-1", tier=SubscriptionTier.MAX, price_credits=0, now=NOW
        )
    with pytest.raises(ValueError):
        await start_card_reset(
            "user-1",
            "key-card-1",
            tier=SubscriptionTier.MAX,
            price_credits=8_000,
            now=NOW,
        )
    table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_grant_stands_when_it_cannot_be_announced(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        usage_reset, "mark_reset", AsyncMock(side_effect=RedisError("down"))
    )

    await announce_reset("user-1")
