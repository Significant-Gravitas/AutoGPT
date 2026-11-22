"""A top-up named by a request key charges at most once for that request, and
the key survives activation so a repeat can find it."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.credit import UserCredit


def _card(pm_id: str) -> MagicMock:
    card = MagicMock()
    card.id = pm_id
    return card


def _succeeded(intent_id: str) -> MagicMock:
    intent = MagicMock()
    intent.status = "succeeded"
    intent.id = intent_id
    intent.amount = 500
    intent.currency = "usd"
    return intent


@contextmanager
def _ledger(credit: UserCredit, *, already_credited: bool):
    with (
        patch(
            "backend.data.credit._top_up_credited_for_request",
            new_callable=AsyncMock,
            return_value=already_credited,
        ) as credited,
        patch.object(
            credit,
            "_add_transaction",
            new_callable=AsyncMock,
            return_value=(0, "tx_pending"),
        ) as add,
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
            return_value=[_card("pm_1")],
        ),
        patch(
            "backend.data.credit.stripe.PaymentIntent.create_async",
            new_callable=AsyncMock,
            return_value=_succeeded("pi_ok"),
        ) as payment_intent,
        patch("backend.data.credit._track_billing_event"),
    ):
        yield credited, add, enable, payment_intent


@pytest.mark.asyncio
async def test_a_request_already_credited_is_not_charged_again():
    credit = UserCredit()
    with _ledger(credit, already_credited=True) as (credited, add, _, intent):
        await credit.top_up_credits("user-1", 500, idempotency_key="click-1")

    credited.assert_awaited_once_with("user-1", "click-1")
    add.assert_not_awaited()
    intent.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_new_request_charges_and_keeps_its_key_through_activation():
    credit = UserCredit()
    with _ledger(credit, already_credited=False) as (_, add, enable, intent):
        await credit.top_up_credits("user-1", 500, idempotency_key="click-1")

    assert add.await_args.kwargs["metadata"].data["idempotency_key"] == "click-1"
    # Activation replaces the row's metadata; the key must be in what it writes.
    assert enable.await_args.kwargs["metadata"].data["idempotency_key"] == "click-1"
    # A retry after a timed-out call gets Stripe's first charge back.
    assert intent.await_args.kwargs["idempotency_key"] == (
        "top-up-user-1-click-1-500-pm_1"
    )


@pytest.mark.asyncio
async def test_a_top_up_without_a_key_charges_as_before():
    credit = UserCredit()
    with _ledger(credit, already_credited=False) as (credited, _, enable, intent):
        await credit.top_up_credits("user-1", 500)

    credited.assert_not_awaited()
    assert "idempotency_key" not in intent.await_args.kwargs
    assert "idempotency_key" not in enable.await_args.kwargs["metadata"].data
