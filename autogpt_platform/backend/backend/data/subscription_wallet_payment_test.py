"""Wallet payment settlement across Stripe API versions, crashes and replays.

Uses the stateful fakes of ``subscription_payment_failure_test``. From
2025-03-31.basil the Invoice has no ``paid_out_of_band``/``payment_intent``,
so these tests prove the wallet's own record decides, not those fields.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import stripe

from backend.data.stripe_invoice_payments import paid_by_stripe_collection
from backend.data.subscription_payment_failure import (
    FailedInvoice,
    handle_subscription_payment_failure,
)
from backend.data.subscription_payment_failure_test import (
    ACACIA,
    CUSTOMER,
    ENDIVE,
    USER,
    World,
    _renewal_failed,
)
from backend.data.subscription_wallet_payment import (
    pay_invoice_from_wallet,
    reconcile_wallet_payment_on_paid_invoice,
)

# (SDK version our calls use, webhook endpoint version of the payloads)
VERSIONS = [(ACACIA, ACACIA), (ENDIVE, ACACIA), (ENDIVE, ENDIVE), (ACACIA, ENDIVE)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version,endpoint_version", VERSIONS)
async def test_success_event_after_a_wallet_payment_never_refunds(
    sdk_version, endpoint_version
):
    """Sentry's finding: without ``paid_out_of_band`` the success event used
    to read a wallet-paid invoice as card-paid and refund it (a free period)."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        paid = world.stripe.view("in_1", endpoint_version)
        await reconcile_wallet_payment_on_paid_invoice(paid)
        await reconcile_wallet_payment_on_paid_invoice(paid)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.balance == 3000
    assert world.ledger.settled("in_1")


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version,endpoint_version", VERSIONS)
async def test_card_paying_an_unsettled_wallet_debit_refunds_it_once(
    sdk_version, endpoint_version
):
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        world.stripe.card_pays("in_1")

        paid = world.stripe.view("in_1", endpoint_version)
        await reconcile_wallet_payment_on_paid_invoice(paid)
        await reconcile_wallet_payment_on_paid_invoice(paid)
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000, "in_1:wallet-refund": 2000}
    assert world.ledger.balance == 5000
    assert world.stripe.paid_out_of_band == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version,endpoint_version", VERSIONS)
async def test_crash_after_stripe_paid_before_settlement_was_recorded(
    sdk_version, endpoint_version
):
    """Stripe accepted the out-of-band pay, then our settlement write died.
    The success event finds the debit unsettled and a paid invoice that no
    card paid: it records the settlement and keeps the money."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        world.ledger.fail_next_update = 1
        with pytest.raises(ConnectionError):
            await handle_subscription_payment_failure(event)
        assert world.stripe.invoices["in_1"]["status"] == "paid"
        assert not world.ledger.settled("in_1")

        await reconcile_wallet_payment_on_paid_invoice(
            world.stripe.view("in_1", endpoint_version)
        )
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.settled("in_1")
    assert world.stripe.paid_out_of_band == ["in_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version", [ACACIA, ENDIVE])
async def test_lost_pay_response_is_settled_not_refunded(sdk_version):
    """The pay reached Stripe but its response was lost: re-reading the
    invoice finds it paid with no card payment, so it settles, not refunds."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay_response"] = 1
        await handle_subscription_payment_failure(event)
        assert world.ledger.settled("in_1")

        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.settled("in_1")


@pytest.mark.asyncio
async def test_paid_invoice_that_does_not_say_who_paid_keeps_the_debit():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        bare = {"id": "in_1", "customer": "cus_1", "status": "paid"}
        with patch.object(
            stripe.Invoice, "retrieve_async", AsyncMock(return_value=bare)
        ):
            assert await paid_by_stripe_collection(bare) is None
            await reconcile_wallet_payment_on_paid_invoice(bare)

    assert world.ledger.transactions == {"in_1": -2000}
    assert not world.ledger.settled("in_1")


@pytest.mark.asyncio
async def test_refunded_debit_never_pays_a_reopened_invoice():
    """CodeRabbit's finding: a refunded debit stayed visible as a wallet
    payment, so a later payable delivery marked the invoice paid out of band
    after the money was already back in the wallet."""
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        # A delivery while the subscription is not unpaid refunds the debit.
        world.stripe.subscriptions["sub_1"]["status"] = "active"
        await handle_subscription_payment_failure(event)
        assert world.ledger.balance == 5000
        # The invoice is payable again and the failure is retried.
        world.stripe.subscriptions["sub_1"]["status"] = "past_due"
        await handle_subscription_payment_failure(event)

    assert world.stripe.paid_out_of_band == []
    assert world.stripe.invoices["in_1"]["status"] == "open"
    assert world.ledger.transactions == {"in_1": -2000, "in_1:wallet-refund": 2000}
    assert world.ledger.balance == 5000


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version", [ACACIA, ENDIVE])
async def test_processing_payment_is_left_alone_before_touching_the_wallet(
    sdk_version,
):
    """Capy's finding: with balance available and a bank debit processing,
    the wallet was debited and Stripe refused the pay on every retry."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world, payment_intent="pi_1")
        world.stripe.payment_intents["pi_1"] = {"id": "pi_1", "status": "processing"}
        await handle_subscription_payment_failure(event)
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {}
    assert world.stripe.paid_out_of_band == []
    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert [s["status"] for s in world.synced] == ["past_due", "past_due"]


@pytest.mark.asyncio
async def test_unfinished_wallet_payment_waits_for_a_processing_payment():
    with World(balance=5000) as world:
        event = _renewal_failed(world, payment_intent="pi_1")
        world.stripe.payment_intents["pi_1"] = {
            "id": "pi_1",
            "status": "requires_payment_method",
        }
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        world.stripe.payment_intents["pi_1"]["status"] = "processing"

        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.stripe.paid_out_of_band == []
    assert world.synced[-1]["status"] == "past_due"


@pytest.mark.asyncio
async def test_concurrent_delivery_without_balance_for_a_second_bill_finishes_it():
    """Two deliveries of one failure: the second finds too little balance
    (checked before the key) and must finish the first's payment, not report
    that the wallet cannot pay, which would cancel the plan being paid for."""
    with World(balance=3000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await pay_invoice_from_wallet("user-1", "cus_1", "sub_1", event)
        assert world.ledger.balance == 1000

        assert await pay_invoice_from_wallet("user-1", "cus_1", "sub_1", event)

    assert world.stripe.paid_out_of_band == ["in_1"]
    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.settled("in_1")
    assert world.stripe.cancelled == []


def _stale_delivery(world: World, stale_invoice: dict):
    """A delivery that read the invoice before another one paid it and the
    subscription after: ``open`` but ``active``, so it will not pay."""
    stale = FailedInvoice(
        user_id=USER,
        customer_id=CUSTOMER,
        invoice=stale_invoice,
        subscription=dict(world.stripe.subscriptions["sub_1"]),
    )
    return patch(
        "backend.data.subscription_payment_failure._load_failed_invoice",
        AsyncMock(return_value=stale),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version", [ACACIA, ENDIVE])
async def test_racing_delivery_with_a_stale_open_invoice_never_refunds(sdk_version):
    """Sentry's race: delivery A pays the invoice; delivery B, holding a stale
    ``open`` copy, found A's debit unsettled and refunded it while Stripe kept
    the invoice paid. B now waits for A's lock and finds the debit settled."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        stale_invoice = world.stripe.view("in_1")
        a_paid, release_a = asyncio.Event(), asyncio.Event()

        async def pause_between_pay_and_settlement():
            a_paid.set()
            await release_a.wait()

        world.stripe.after_pay = pause_between_pay_and_settlement
        a = asyncio.create_task(handle_subscription_payment_failure(event))
        await a_paid.wait()
        world.stripe.after_pay = None
        with _stale_delivery(world, stale_invoice):
            b = asyncio.create_task(handle_subscription_payment_failure(event))
            for _ in range(50):
                await asyncio.sleep(0)
            assert not b.done()
            release_a.set()
            await asyncio.gather(a, b)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.settled("in_1")
    assert world.stripe.paid_out_of_band == ["in_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_version", [ACACIA, ENDIVE])
async def test_stale_open_copy_after_an_unrecorded_wallet_pay_never_refunds(
    sdk_version,
):
    """A paid at Stripe and died before recording it; B arrives with a stale
    ``open`` copy. The decision re-reads the invoice, finds it paid with no
    card payment, and settles instead of refunding."""
    with World(balance=5000, api_version=sdk_version) as world:
        event = _renewal_failed(world)
        stale_invoice = world.stripe.view("in_1")
        world.ledger.fail_next_update = 1
        with pytest.raises(ConnectionError):
            await handle_subscription_payment_failure(event)
        with _stale_delivery(world, stale_invoice):
            await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.settled("in_1")
