"""A declined saved card moves on to the next one, and says why when all fail."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe

from backend.data.credit import CardDeclinedError, UserCredit


def _card(pm_id: str) -> MagicMock:
    card = MagicMock()
    card.id = pm_id
    return card


def _declined(message: str = "Your card has insufficient funds.") -> stripe.CardError:
    return stripe.CardError(
        message,
        None,
        "card_declined",
        http_status=402,
        headers={"request-id": "req_secret123"},
    )


def _succeeded(intent_id: str) -> MagicMock:
    intent = MagicMock()
    intent.status = "succeeded"
    intent.id = intent_id
    intent.amount = 500
    intent.currency = "usd"
    return intent


@contextmanager
def _stripe(credit: UserCredit, cards: list[MagicMock], intent_results: list):
    with (
        patch.object(
            credit,
            "_add_transaction",
            new_callable=AsyncMock,
            return_value=(0, "tx_pending"),
        ),
        patch.object(
            credit,
            "_enable_transaction",
            new_callable=AsyncMock,
            return_value=(500, "pi_ok"),
        ) as enable,
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit.stripe.PaymentMethod.list_async",
            new_callable=AsyncMock,
            return_value=cards,
        ),
        patch(
            "backend.data.credit.stripe.PaymentIntent.create_async",
            new_callable=AsyncMock,
            side_effect=intent_results,
        ) as payment_intent,
        patch(
            "backend.data.credit.stripe.SetupIntent.create_async",
            new_callable=AsyncMock,
            side_effect=intent_results,
        ) as setup_intent,
        patch("backend.data.credit._track_billing_event"),
    ):
        yield enable, payment_intent, setup_intent


@pytest.mark.asyncio
async def test_a_single_declined_card_raises_stripes_message():
    credit = UserCredit()
    with _stripe(credit, [_card("pm_1")], [_declined()]) as (enable, *_):
        with pytest.raises(CardDeclinedError) as exc:
            await credit._top_up_credits("user-1", 500)

    assert str(exc.value) == "Your card has insufficient funds."
    assert "req_" not in str(exc.value)
    enable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_declined_card_moves_on_to_the_next_saved_card():
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2")]
    with _stripe(credit, cards, [_declined(), _succeeded("pi_ok")]) as (
        enable,
        payment_intent,
        _,
    ):
        await credit._top_up_credits("user-1", 500)

    tried = [c.kwargs["payment_method"] for c in payment_intent.await_args_list]
    assert tried == ["pm_1", "pm_2"]
    enable.assert_awaited_once()
    assert enable.await_args.kwargs["new_transaction_key"] == "pi_ok"


@pytest.mark.asyncio
async def test_every_card_declined_raises_the_last_decline_and_saves_nothing():
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2")]
    declines = [_declined("Your card was declined."), _declined()]
    with _stripe(credit, cards, declines) as (enable, payment_intent, _):
        with pytest.raises(CardDeclinedError) as exc:
            await credit._top_up_credits("user-1", 500)

    assert payment_intent.await_count == 2
    assert "Your card has insufficient funds." in str(exc.value)
    assert "req_" not in str(exc.value)
    enable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_declined_card_check_raises_stripes_message():
    credit = UserCredit()
    with _stripe(credit, [_card("pm_1")], [_declined()]) as (enable, _, setup):
        with pytest.raises(CardDeclinedError) as exc:
            await credit._top_up_credits("user-1", 0)

    setup.assert_awaited_once()
    assert str(exc.value) == "Your card has insufficient funds."
    enable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_decline_without_a_message_still_says_the_card_was_declined():
    credit = UserCredit()
    with _stripe(credit, [_card("pm_1")], [_declined("")]):
        with pytest.raises(CardDeclinedError) as exc:
            await credit._top_up_credits("user-1", 500)

    assert str(exc.value) == "Your card was declined."


@pytest.mark.asyncio
async def test_a_decline_asking_for_bank_approval_is_flagged():
    credit = UserCredit()
    sca = stripe.CardError(
        "Your card was declined. This transaction requires authentication.",
        None,
        "authentication_required",
        http_status=402,
    )
    with _stripe(credit, [_card("pm_1")], [sca]):
        with pytest.raises(CardDeclinedError) as exc:
            await credit._top_up_credits("user-1", 500)

    assert exc.value.needs_authentication
