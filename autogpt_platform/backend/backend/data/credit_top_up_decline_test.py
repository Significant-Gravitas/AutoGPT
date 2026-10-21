"""A declined saved card moves on to the next one, except on an automatic
top-up, and says why when the top-up fails."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe

from backend.data.credit import CardDeclinedError, UserCredit
from backend.data.model import TopUpType


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


def _hard_declined() -> stripe.CardError:
    message = "Your card was declined."
    return stripe.CardError(
        message,
        None,
        "card_declined",
        http_status=402,
        json_body={
            "error": {
                "type": "card_error",
                "code": "card_declined",
                "decline_code": "stolen_card",
                "message": message,
            }
        },
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
    with (
        _stripe(credit, cards, declines) as (enable, payment_intent, _),
        pytest.raises(CardDeclinedError) as exc,
    ):
        await credit._top_up_credits("user-1", 500)

    assert payment_intent.await_count == 2
    assert "Your card has insufficient funds." in str(exc.value)
    assert "req_" not in str(exc.value)
    enable.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_decline_stripe_sends_without_a_decline_code_moves_on():
    # Stripe leaves decline_code out of the error body for an expired card;
    # its ErrorObject fills the gap with None, so reading it cannot raise.
    message = "Your card has expired."
    expired = stripe.CardError(
        message,
        None,
        "expired_card",
        http_status=402,
        json_body={
            "error": {"type": "card_error", "code": "expired_card", "message": message}
        },
    )
    assert expired.error is not None
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2")]
    with _stripe(credit, cards, [expired, _succeeded("pi_ok")]) as (
        enable,
        payment_intent,
        _,
    ):
        await credit._top_up_credits("user-1", 500)

    tried = [c.kwargs["payment_method"] for c in payment_intent.await_args_list]
    assert tried == ["pm_1", "pm_2"]
    enable.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_decline_without_a_message_still_says_the_card_was_declined():
    credit = UserCredit()
    with (
        _stripe(credit, [_card("pm_1")], [_declined("")]),
        pytest.raises(CardDeclinedError) as exc,
    ):
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
    with (
        _stripe(credit, [_card("pm_1")], [sca]),
        pytest.raises(CardDeclinedError) as exc,
    ):
        await credit._top_up_credits("user-1", 500)

    assert exc.value.needs_authentication


@pytest.mark.asyncio
async def test_an_automatic_top_up_stops_at_the_first_declined_card():
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2"), _card("pm_3")]
    results = [_hard_declined(), _succeeded("pi_2"), _succeeded("pi_3")]
    with (
        _stripe(credit, cards, results) as (enable, payment_intent, _),
        pytest.raises(CardDeclinedError) as exc,
    ):
        await credit._top_up_credits("user-1", 500, top_up_type=TopUpType.AUTO)

    tried = [c.kwargs["payment_method"] for c in payment_intent.await_args_list]
    assert tried == ["pm_1"]
    assert str(exc.value) == "Your card was declined."
    enable.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_automatic_top_up_needing_bank_approval_stops_and_is_flagged():
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2")]
    sca = stripe.CardError(
        "Your card was declined. This transaction requires authentication.",
        None,
        "authentication_required",
        http_status=402,
    )
    with (
        _stripe(credit, cards, [sca, _succeeded("pi_2")]) as (_, payment_intent, _),
        pytest.raises(CardDeclinedError) as exc,
    ):
        await credit._top_up_credits("user-1", 500, top_up_type=TopUpType.AUTO)

    assert payment_intent.await_count == 1
    assert exc.value.needs_authentication


@pytest.mark.asyncio
@pytest.mark.parametrize("top_up_type", [TopUpType.MANUAL, TopUpType.UNCATEGORIZED])
async def test_a_top_up_the_user_starts_still_tries_the_next_card(top_up_type):
    credit = UserCredit()
    cards = [_card("pm_1"), _card("pm_2"), _card("pm_3")]
    results = [_hard_declined(), _succeeded("pi_ok")]
    with _stripe(credit, cards, results) as (enable, payment_intent, _):
        await credit._top_up_credits("user-1", 500, top_up_type=top_up_type)

    tried = [c.kwargs["payment_method"] for c in payment_intent.await_args_list]
    assert tried == ["pm_1", "pm_2"]
    enable.assert_awaited_once()
