"""A paid or irrelevant invoice must not generate a payment-failure notice."""

import pytest
import stripe

from backend.data.subscription_payment_failure import (
    handle_subscription_payment_failure,
)
from backend.data.subscription_payment_failure_test import (
    World,
    _renewal_failed,
    _superseded_by_a_newer_plan,
)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_status", ["unpaid", "past_due"])
async def test_invoice_stripe_marked_uncollectible_still_requires_a_notice(sub_status):
    """Stripe can mark the invoice uncollectible after its last retry; its
    payment link still works, so the final notice must go out."""
    with World(balance=5000) as world:
        invoice = _renewal_failed(world, status="uncollectible")
        world.stripe.subscriptions["sub_1"]["status"] = sub_status
        outcome = await handle_subscription_payment_failure(invoice)

    assert outcome == "needs_action"
    assert world.ledger.balance == 5000
    assert world.stripe.paid_out_of_band == []
    assert world.stripe.cancelled == []
    assert world.stripe.voided == []


@pytest.mark.asyncio
async def test_wallet_debit_refunded_from_an_uncollectible_invoice_requires_a_notice():
    """The wallet does not pay an uncollectible invoice; it refunds an unfinished
    debit, and the customer is told in that same delivery."""
    with World(balance=5000) as world:
        invoice = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(invoice)
        world.stripe.invoices["in_1"]["status"] = "uncollectible"
        world.stripe.subscriptions["sub_1"]["status"] = "unpaid"
        outcome = await handle_subscription_payment_failure(invoice)

    assert outcome == "needs_action"
    assert world.ledger.balance == 5000
    assert world.stripe.paid_out_of_band == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sub_status,amount_due,latest",
    [("canceled", 2000, "in_1"), ("unpaid", 0, "in_1"), ("unpaid", 2000, "in_2")],
)
async def test_uncollectible_invoice_nobody_must_pay_requires_no_notice(
    sub_status, amount_due, latest
):
    with World() as world:
        invoice = _renewal_failed(world, status="uncollectible", amount_due=amount_due)
        world.stripe.subscriptions["sub_1"]["status"] = sub_status
        world.stripe.subscriptions["sub_1"]["latest_invoice"] = latest
        outcome = await handle_subscription_payment_failure(invoice)

    assert outcome == "ignored"


@pytest.mark.asyncio
async def test_uncollectible_invoice_of_a_replaced_plan_requires_no_notice():
    """Paying it would reactivate the plan the customer replaced."""
    with World() as world:
        invoice = _superseded_by_a_newer_plan(world, balance=0)
        world.stripe.invoices["in_old"]["status"] = "uncollectible"
        world.stripe.subscriptions["sub_old"]["status"] = "unpaid"
        outcome = await handle_subscription_payment_failure(invoice)

    assert outcome == "ignored"
    assert world.stripe.cancelled == []
