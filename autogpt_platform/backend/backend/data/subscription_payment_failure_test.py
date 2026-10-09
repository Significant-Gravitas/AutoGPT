"""Failed-payment handling against a fake Stripe account and wallet ledger.

The fakes hold state, so a test can deliver the same event twice, fail a
Stripe call part way, and check what a retry does with what was left behind.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.errors import UniqueViolationError

from backend.data.credit import PAYMENT_FAILURE_CANCELLATION_COMMENT
from backend.data.subscription_payment_failure import (
    handle_subscription_payment_failure,
)
from backend.data.subscription_wallet_payment import (
    refund_wallet_debit_if_paid_by_card,
)
from backend.util.exceptions import InsufficientBalanceError

CUSTOMER = "cus_1"
USER = "user-1"


class FakeStripe:
    def __init__(self) -> None:
        self.invoices: dict[str, dict] = {}
        self.subscriptions: dict[str, dict] = {}
        self.payment_intents: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self.voided: list[str] = []
        self.paid_out_of_band: list[str] = []
        self.fail_next: dict[str, int] = {}

    def add_subscription(self, sub_id: str, status: str, latest_invoice: str = ""):
        self.subscriptions[sub_id] = {
            "id": sub_id,
            "customer": CUSTOMER,
            "status": status,
            "latest_invoice": latest_invoice or None,
            "metadata": {},
            "items": {"data": [{"price": {"id": "price_pro"}}]},
        }

    def add_invoice(
        self,
        invoice_id: str,
        sub_id: str,
        status: str = "open",
        amount_due: int = 2000,
        payment_intent: str | None = None,
    ):
        self.invoices[invoice_id] = {
            "id": invoice_id,
            "customer": CUSTOMER,
            "subscription": sub_id,
            "status": status,
            "amount_due": amount_due,
            "paid_out_of_band": False,
            "payment_intent": payment_intent,
        }
        return dict(self.invoices[invoice_id])

    def _maybe_fail(self, call: str) -> None:
        if self.fail_next.get(call, 0) > 0:
            self.fail_next[call] -= 1
            raise stripe.APIConnectionError(f"{call} failed")

    async def retrieve_invoice(self, invoice_id: str):
        return dict(self.invoices[invoice_id])

    async def retrieve_subscription(self, sub_id: str):
        return dict(self.subscriptions[sub_id])

    async def retrieve_payment_intent(self, pi_id: str):
        return dict(self.payment_intents[pi_id])

    async def pay(self, invoice_id: str, paid_out_of_band: bool = False):
        self._maybe_fail("pay")
        invoice = self.invoices[invoice_id]
        if invoice["status"] not in ("open", "uncollectible"):
            raise stripe.InvalidRequestError("Invoice is already paid", None)
        invoice.update(status="paid", paid_out_of_band=paid_out_of_band)
        self.paid_out_of_band.append(invoice_id)
        sub = self.subscriptions[invoice["subscription"]]
        if sub["latest_invoice"] == invoice_id:
            sub["status"] = "active"
        return dict(invoice)

    async def cancel(self, sub_id: str, **params):
        self._maybe_fail("cancel")
        sub = self.subscriptions[sub_id]
        if sub["status"] == "canceled":
            raise stripe.InvalidRequestError("already canceled", None)
        assert params == {
            "cancellation_details": {"comment": PAYMENT_FAILURE_CANCELLATION_COMMENT}
        }
        sub["status"] = "canceled"
        self.cancelled.append(sub_id)
        return dict(sub)

    async def list_invoices(self, subscription: str, status: str, limit: int):
        page = MagicMock()
        page.data = [
            stripe.Invoice.construct_from(dict(inv), "sk_test")
            for inv in self.invoices.values()
            if inv["subscription"] == subscription and inv["status"] == status
        ]
        page.has_more = False
        return page

    async def void(self, invoice_id: str):
        self._maybe_fail("void")
        invoice = self.invoices[invoice_id]
        if invoice["status"] != "open":
            raise stripe.InvalidRequestError("not open", None)
        invoice["status"] = "void"
        self.voided.append(invoice_id)
        return dict(invoice)


class FakeLedger:
    """The wallet: a balance plus transactions unique by key, like the DB."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.transactions: dict[str, int] = {}

    async def add_transaction(self, *, user_id, amount, transaction_key, **kwargs):
        if transaction_key in self.transactions:
            raise UniqueViolationError({"error": "duplicate key"})
        if kwargs.get("fail_insufficient_credits") and self.balance + amount < 0:
            raise InsufficientBalanceError(
                message="no balance",
                user_id=user_id,
                balance=self.balance,
                amount=amount,
            )
        self.transactions[transaction_key] = amount
        self.balance += amount
        return self.balance, transaction_key

    async def find_unique(self, where):
        key = where["creditTransactionIdentifier"]["transactionKey"]
        if key not in self.transactions:
            return None
        return MagicMock(amount=self.transactions[key])


class World:
    def __init__(self, balance: int = 0, trial=None) -> None:
        self.stripe = FakeStripe()
        self.ledger = FakeLedger(balance)
        self.synced: list[dict] = []
        self.trial = trial
        self._stack = ExitStack()

    async def _sync(self, subscription: dict, **kwargs):
        self.synced.append(dict(subscription))

    def __enter__(self):
        user = MagicMock(id=USER, subscriptionTier=SubscriptionTier.PRO)
        users = MagicMock(find_first=AsyncMock(return_value=user))
        fs = self.stripe
        patches = [
            patch(
                "backend.data.subscription_payment_failure.User.prisma",
                return_value=users,
            ),
            patch(
                "backend.data.subscription_wallet_payment.User.prisma",
                return_value=users,
            ),
            patch(
                "backend.data.subscription_wallet_payment.CreditTransaction.prisma",
                return_value=MagicMock(find_unique=self.ledger.find_unique),
            ),
            patch(
                "backend.data.subscription_wallet_payment.UserCredit._add_transaction",
                side_effect=self.ledger.add_transaction,
            ),
            patch(
                "backend.data.subscription_payment_failure.sync_subscription_from_stripe",
                side_effect=self._sync,
            ),
            patch(
                "backend.data.subscription_payment_failure.get_subscription_trial",
                new=AsyncMock(return_value=self.trial),
            ),
            patch.object(stripe.Invoice, "retrieve_async", fs.retrieve_invoice),
            patch.object(stripe.Invoice, "pay_async", fs.pay),
            patch.object(stripe.Invoice, "list_async", fs.list_invoices),
            patch.object(stripe.Invoice, "void_invoice_async", fs.void),
            patch.object(stripe.Subscription, "retrieve_async", fs.retrieve_subscription),
            patch.object(stripe.Subscription, "cancel_async", fs.cancel),
            patch.object(
                stripe.PaymentIntent, "retrieve_async", fs.retrieve_payment_intent
            ),
        ]
        for p in patches:
            self._stack.enter_context(p)
        return self

    def __exit__(self, *exc):
        self._stack.close()


def _renewal_failed(world: World, sub_id="sub_1", invoice_id="in_1", **kw) -> dict:
    world.stripe.add_subscription(sub_id, "past_due", latest_invoice=invoice_id)
    return world.stripe.add_invoice(invoice_id, sub_id, **kw)


@pytest.mark.asyncio
async def test_unpaid_past_due_subscription_is_cancelled_and_its_invoice_voided():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_1"]
    assert world.stripe.voided == ["in_1"]
    assert world.synced[-1]["status"] == "canceled"
    assert world.ledger.transactions == {}


@pytest.mark.asyncio
async def test_old_failure_never_cancels_a_newer_active_subscription():
    """The reproduced defect: an old past-due subscription's failed retry
    used to cancel the customer's newer, paid subscription instead."""
    with World(balance=0) as world:
        event = _renewal_failed(world, sub_id="sub_old", invoice_id="in_old")
        world.stripe.add_subscription("sub_new", "active", latest_invoice="in_new")
        world.stripe.add_invoice("in_new", "sub_new", status="paid")
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_old"]
    assert world.stripe.subscriptions["sub_new"]["status"] == "active"
    assert world.stripe.voided == ["in_old"]
    # The tier is recomputed from the cancelled subscription, so the sync can
    # see sub_new and keep the paid tier, instead of a blanket NO_TIER write.
    assert [s["id"] for s in world.synced] == ["sub_old"]


@pytest.mark.asyncio
async def test_failure_of_an_invoice_that_is_no_longer_latest_does_nothing():
    with World(balance=5000) as world:
        world.stripe.add_subscription("sub_1", "past_due", latest_invoice="in_2")
        event = world.stripe.add_invoice("in_1", "sub_1")
        world.stripe.add_invoice("in_2", "sub_1")
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.paid_out_of_band == []
    assert world.ledger.transactions == {}


@pytest.mark.asyncio
async def test_delayed_failure_after_the_customer_paid_does_nothing():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        world.stripe.invoices["in_1"]["status"] = "paid"
        world.stripe.subscriptions["sub_1"]["status"] = "active"
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert world.synced == []


@pytest.mark.asyncio
async def test_duplicate_failure_event_after_cancellation_is_a_no_op():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_1"]
    assert world.stripe.voided == ["in_1"]


@pytest.mark.asyncio
async def test_void_failure_is_resumed_by_the_retry():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["void"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        assert world.stripe.cancelled == ["sub_1"]
        assert world.stripe.invoices["in_1"]["status"] == "open"

        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_1"]
    assert world.stripe.voided == ["in_1"]


@pytest.mark.asyncio
async def test_cancel_failure_raises_so_the_webhook_retries():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["cancel"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        assert world.stripe.voided == []
        assert world.synced == []

        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_1"]
    assert world.stripe.voided == ["in_1"]


@pytest.mark.asyncio
async def test_payment_still_processing_is_not_cancelled_or_voided():
    with World(balance=0) as world:
        event = _renewal_failed(world, payment_intent="pi_1")
        world.stripe.payment_intents["pi_1"] = {"id": "pi_1", "status": "processing"}
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert [s["status"] for s in world.synced] == ["past_due"]


@pytest.mark.asyncio
async def test_unconverted_trial_invoice_stays_open_for_card_repair():
    trial = MagicMock(converted_at=None, subscription_id="sub_1")
    with World(balance=0, trial=trial) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert [s["status"] for s in world.synced] == ["past_due"]


@pytest.mark.asyncio
async def test_converted_trial_renewal_failure_is_cancelled():
    trial = MagicMock(
        converted_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        subscription_id="sub_1",
    )
    with World(balance=0, trial=trial) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == ["sub_1"]


@pytest.mark.asyncio
async def test_incomplete_subscription_is_left_alone():
    with World(balance=0) as world:
        world.stripe.add_subscription("sub_1", "incomplete", latest_invoice="in_1")
        event = world.stripe.add_invoice("in_1", "sub_1")
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.synced == []


@pytest.mark.asyncio
async def test_enterprise_user_is_skipped():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        user = MagicMock(id=USER, subscriptionTier=SubscriptionTier.ENTERPRISE)
        with patch(
            "backend.data.subscription_payment_failure.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ):
            await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []


@pytest.mark.asyncio
async def test_invoice_of_another_customer_is_ignored():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.subscriptions["sub_1"]["customer"] = "cus_other"
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.ledger.transactions == {}


@pytest.mark.asyncio
async def test_wallet_covers_invoice_and_marks_it_paid_out_of_band():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.balance == 3000
    assert world.stripe.invoices["in_1"]["paid_out_of_band"] is True
    assert world.stripe.cancelled == []


@pytest.mark.asyncio
async def test_duplicate_failure_event_debits_the_wallet_once():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.balance == 3000
    assert world.stripe.paid_out_of_band == ["in_1"]


@pytest.mark.asyncio
async def test_wallet_pay_failure_keeps_debit_and_the_retry_finishes_it():
    """Defect 5: the debit used to stay while Stripe still saw the bill unpaid,
    and the retry crashed on the debit's unique key without paying."""
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        assert world.ledger.transactions == {"in_1": -2000}
        assert world.stripe.invoices["in_1"]["status"] == "open"

        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.ledger.balance == 3000
    assert world.stripe.invoices["in_1"]["status"] == "paid"
    assert world.stripe.cancelled == []


@pytest.mark.asyncio
async def test_card_paying_first_refunds_the_unfinished_wallet_debit():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        # Stripe's own card retry succeeds before the webhook retry arrives.
        world.stripe.invoices["in_1"]["status"] = "paid"
        world.stripe.subscriptions["sub_1"]["status"] = "active"

        await refund_wallet_debit_if_paid_by_card(
            dict(world.stripe.invoices["in_1"])
        )
        await handle_subscription_payment_failure(event)

    assert world.ledger.balance == 5000
    assert world.ledger.transactions == {"in_1": -2000, "in_1:wallet-refund": 2000}
    assert world.stripe.paid_out_of_band == []


@pytest.mark.asyncio
async def test_resumed_wallet_payment_is_refunded_once_the_plan_is_gone():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        world.stripe.subscriptions["sub_1"]["status"] = "canceled"

        await handle_subscription_payment_failure(event)

    assert world.ledger.balance == 5000
    assert world.stripe.paid_out_of_band == []


@pytest.mark.asyncio
async def test_wallet_paid_invoice_is_not_refunded_on_its_success_event():
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        await refund_wallet_debit_if_paid_by_card(
            dict(world.stripe.invoices["in_1"])
        )

    assert world.ledger.balance == 3000
    assert "in_1:wallet-refund" not in world.ledger.transactions


@pytest.mark.asyncio
async def test_non_subscription_invoice_is_ignored():
    with World(balance=5000) as world:
        await handle_subscription_payment_failure(
            {"id": "in_x", "customer": CUSTOMER, "amount_due": 2000}
        )

    assert world.ledger.transactions == {}
