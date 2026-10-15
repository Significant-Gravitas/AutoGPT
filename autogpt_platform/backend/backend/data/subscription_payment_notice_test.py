"""A paid or irrelevant invoice must not generate a payment-failure notice."""

import pytest

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
