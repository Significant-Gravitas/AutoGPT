"""A paid or irrelevant invoice must not generate a payment-failure notice."""

import pytest
import stripe

from backend.data.subscription_payment_failure import (
    handle_subscription_payment_failure,
)
from backend.data.subscription_payment_failure_test import World, _renewal_failed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balance,status,expected",
    [(5000, "open", "settled"), (0, "open", "needs_action"), (0, "paid", "ignored")],
)
async def test_only_unpaid_uncovered_invoice_requires_a_notice(
    balance, status, expected
):
    with World(balance=balance) as world:
        invoice = _renewal_failed(world)
        world.stripe.invoices["in_1"]["status"] = status
        outcome = await handle_subscription_payment_failure(invoice)
    assert outcome == expected


@pytest.mark.asyncio
async def test_unpaid_invoice_whose_wallet_debit_was_refunded_requires_a_notice():
    with World(balance=5000) as world:
        invoice = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(invoice)
        status = world.stripe.subscriptions["sub_1"]["status"]
        world.stripe.subscriptions["sub_1"]["status"] = "canceled"
        await handle_subscription_payment_failure(invoice)  # refunds the debit
        # The subscription is unpaid again, and the refunded debit must never pay it.
        world.stripe.subscriptions["sub_1"]["status"] = status
        outcome = await handle_subscription_payment_failure(invoice)

    assert outcome == "needs_action"
    assert world.ledger.balance == 5000
    assert world.stripe.paid_out_of_band == []
