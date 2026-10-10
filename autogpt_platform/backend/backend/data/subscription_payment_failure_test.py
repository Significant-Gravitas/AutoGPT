"""Failed-payment handling against a fake Stripe account and wallet ledger.

The fakes hold state, so a test can deliver the same event twice, fail a
Stripe call part way, and check what a retry does with what was left behind.
The tier sync is the real one, so a test sees the tier the customer ends up on.
Fake Stripe shapes each invoice by API version like the real one: before
2025-03-31.basil it carries ``paid_out_of_band`` and ``payment_intent``; from
basil on it has neither, and ``payments`` only when expanded.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import CreditTransactionType, SubscriptionTier
from prisma.errors import UniqueViolationError

from backend.data.credit import sync_subscription_from_stripe
from backend.data.subscription_payment_failure import (
    handle_subscription_payment_failure,
)
from backend.data.subscription_wallet_payment import (
    reconcile_wallet_payment_on_paid_invoice,
)
from backend.util.exceptions import InsufficientBalanceError

CUSTOMER = "cus_1"
USER = "user-1"
ACACIA = "2025-02-24.acacia"
ENDIVE = "2026-09-30.endive"


def _is_basil(api_version: str) -> bool:
    return api_version >= "2025-03-31"


class FakeStripe:
    def __init__(self) -> None:
        self.invoices: dict[str, dict] = {}
        self.subscriptions: dict[str, dict] = {}
        self.payment_intents: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self.voided: list[str] = []
        self.last_status: dict[str, str] = {}
        self.paid_out_of_band: list[str] = []
        self.fail_next: dict[str, int] = {}
        # Runs after Stripe applied a pay, before the response returns.
        self.after_pay: Callable[[], Awaitable[None]] | None = None

    def add_subscription(
        self,
        sub_id: str,
        status: str,
        latest_invoice: str = "",
        price: str = "price_pro",
    ):
        self.subscriptions[sub_id] = {
            "id": sub_id,
            "customer": CUSTOMER,
            "status": status,
            "latest_invoice": latest_invoice or None,
            "metadata": {},
            "items": {"data": [{"price": {"id": price}}]},
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
            "_payments": (
                [_invoice_payment("open", payment_intent)] if payment_intent else []
            ),
        }
        return self.view(invoice_id)

    def view(self, invoice_id: str, api_version: str | None = None, expand=()) -> dict:
        """The invoice as Stripe returns it at ``api_version`` (the SDK's by
        default; pass the endpoint's for a webhook payload)."""
        invoice = dict(self.invoices[invoice_id])
        payments = invoice.pop("_payments")
        if not _is_basil(api_version or stripe.api_version):
            return invoice
        del invoice["paid_out_of_band"], invoice["payment_intent"]
        if "payments" in expand:
            invoice["payments"] = {"data": [dict(p) for p in payments]}
        return invoice

    def card_pays(self, invoice_id: str, pi_id: str = "pi_card") -> None:
        """Stripe's own retry of the customer's card succeeds."""
        invoice = self.invoices[invoice_id]
        invoice.update(status="paid", paid_out_of_band=False, payment_intent=pi_id)
        invoice["_payments"].append(_invoice_payment("paid", pi_id))
        sub = self.subscriptions[invoice["subscription"]]
        if sub["latest_invoice"] == invoice_id:
            sub["status"] = "active"

    def card_fails(self, invoice_id: str) -> None:
        """Stripe's own retry of the customer's card fails again."""
        invoice = self.invoices[invoice_id]
        assert invoice["status"] == "open"
        self.subscriptions[invoice["subscription"]]["status"] = "past_due"

    def _maybe_fail(self, call: str) -> None:
        if self.fail_next.get(call, 0) > 0:
            self.fail_next[call] -= 1
            raise stripe.APIConnectionError(f"{call} failed")

    async def retrieve_invoice(self, invoice_id: str, expand=()):
        return self.view(invoice_id, expand=expand)

    async def retrieve_subscription(self, sub_id: str):
        return dict(self.subscriptions[sub_id])

    async def retrieve_payment_intent(self, pi_id: str):
        return dict(self.payment_intents[pi_id])

    async def pay(self, invoice_id: str, paid_out_of_band: bool = False):
        self._maybe_fail("pay")
        invoice = self.invoices[invoice_id]
        if invoice["status"] not in ("open", "uncollectible"):
            raise stripe.InvalidRequestError("Invoice is already paid", None)
        pi = self.payment_intents.get(invoice["payment_intent"] or "")
        if pi and pi["status"] == "processing":
            raise stripe.InvalidRequestError("A payment is in progress", None)
        invoice.update(status="paid", paid_out_of_band=paid_out_of_band)
        self.paid_out_of_band.append(invoice_id)
        sub = self.subscriptions[invoice["subscription"]]
        if sub["latest_invoice"] == invoice_id:
            sub["status"] = "active"
        if self.after_pay:
            await self.after_pay()
        # The request reached Stripe, but its response was lost.
        self._maybe_fail("pay_response")
        return self.view(invoice_id)

    async def cancel(self, sub_id: str, **params):
        sub = self.subscriptions[sub_id]
        if sub["status"] == "canceled":
            raise stripe.InvalidRequestError("already canceled", None)
        sub["status"] = "canceled"
        self.cancelled.append(sub_id)
        return dict(sub)

    async def void(self, invoice_id: str, **params):
        self.voided.append(invoice_id)
        raise AssertionError("an unpaid invoice must stay payable")

    async def list_subscriptions(self, customer: str, status: str, limit: int):
        page = MagicMock()
        page.data = [
            dict(sub)
            for sub in self.subscriptions.values()
            if sub["customer"] == customer and sub["status"] == status
        ]
        page.has_more = False
        return page


def _invoice_payment(status: str, pi_id: str) -> dict:
    return {
        "object": "invoice_payment",
        "status": status,
        "payment": {"type": "payment_intent", "payment_intent": pi_id},
    }


class FakeLedger:
    """The wallet: a balance plus transactions unique by key, like the DB,
    which checks the balance before the key."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.transactions: dict[str, int] = {}
        self.metadata: dict[str, dict] = {}
        self.fail_next_update = 0

    async def add_transaction(self, *, user_id, amount, transaction_key, **kwargs):
        if kwargs.get("fail_insufficient_credits") and self.balance + amount < 0:
            raise InsufficientBalanceError(
                message="no balance",
                user_id=user_id,
                balance=self.balance,
                amount=amount,
            )
        if transaction_key in self.transactions:
            raise UniqueViolationError({"error": "duplicate key"})
        self.transactions[transaction_key] = amount
        self.metadata[transaction_key] = dict(kwargs["metadata"].data)
        self.balance += amount
        return self.balance, transaction_key

    async def find_many(self, where):
        return [
            MagicMock(
                transactionKey=key,
                amount=self.transactions[key],
                type=CreditTransactionType.SUBSCRIPTION,
                metadata=dict(self.metadata[key]),
            )
            for key in where["transactionKey"]["in"]
            if key in self.transactions
        ]

    async def update(self, where, data):
        if self.fail_next_update > 0:
            self.fail_next_update -= 1
            raise ConnectionError("database went away")
        key = where["creditTransactionIdentifier"]["transactionKey"]
        self.metadata[key] = dict(data["metadata"].data)

    def settled(self, invoice_id: str) -> bool:
        return self.metadata.get(invoice_id, {}).get("wallet_payment") == "settled"


class World:
    def __init__(self, balance: int = 0, api_version: str = ACACIA) -> None:
        self.api_version = api_version
        self.stripe = FakeStripe()
        self.ledger = FakeLedger(balance)
        self.synced: list[dict] = []
        self.tier = SubscriptionTier.PRO
        self.tier_writes: list[SubscriptionTier] = []
        self.locks: dict[str, asyncio.Lock] = {}
        self._stack = ExitStack()

    @asynccontextmanager
    async def _lock(self, invoice_id: str):
        async with self.locks.setdefault(invoice_id, asyncio.Lock()):
            yield

    async def _sync(self, subscription: dict, **kwargs):
        self.synced.append(dict(subscription))
        await sync_subscription_from_stripe(subscription, **kwargs)

    async def _find_user(self, **kwargs):
        return MagicMock(id=USER, subscriptionTier=self.tier)

    async def _set_tier(self, user_id: str, tier: SubscriptionTier, **kwargs):
        self.tier = tier
        self.tier_writes.append(tier)

    def subscription_updated(self, sub_id: str) -> Awaitable[None]:
        """Stripe's ``customer.subscription.updated`` for ``sub_id``."""
        return sync_subscription_from_stripe(dict(self.stripe.subscriptions[sub_id]))

    def __enter__(self):
        users = MagicMock(find_first=self._find_user)
        fs = self.stripe
        patches = [
            patch("backend.data.credit.User.prisma", return_value=users),
            patch("backend.data.credit.set_subscription_tier", self._set_tier),
            patch(
                "backend.data.credit.build_price_to_tier_map",
                new=AsyncMock(
                    return_value={
                        "price_pro": SubscriptionTier.PRO,
                        "price_max": SubscriptionTier.MAX,
                    }
                ),
            ),
            patch("backend.data.credit._track_billing_event"),
            patch("backend.data.credit.schedule_posthog_lifecycle_sync"),
            patch(
                "backend.data.credit.get_pending_subscription_change",
                new=MagicMock(),
            ),
            patch.object(stripe.Subscription, "list_async", fs.list_subscriptions),
            patch(
                "backend.data.subscription_payment_failure.User.prisma",
                return_value=users,
            ),
            patch(
                "backend.data.subscription_wallet_payment.User.prisma",
                return_value=users,
            ),
            patch(
                "backend.data.wallet_payment_state.CreditTransaction.prisma",
                return_value=MagicMock(
                    find_many=self.ledger.find_many, update=self.ledger.update
                ),
            ),
            patch.object(stripe, "api_version", self.api_version),
            patch(
                "backend.data.subscription_wallet_payment._wallet_payment_lock",
                self._lock,
            ),
            patch(
                "backend.data.subscription_wallet_payment.UserCredit._add_transaction",
                side_effect=self.ledger.add_transaction,
            ),
            patch(
                "backend.data.subscription_payment_failure.sync_subscription_from_stripe",
                side_effect=self._sync,
            ),
            patch.object(stripe.Invoice, "retrieve_async", fs.retrieve_invoice),
            patch.object(stripe.Invoice, "pay_async", fs.pay),
            patch.object(stripe.Invoice, "void_invoice_async", fs.void),
            patch.object(
                stripe.Subscription, "retrieve_async", fs.retrieve_subscription
            ),
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
async def test_unpaid_renewal_cuts_access_and_leaves_the_bill_payable():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)

    assert world.tier == SubscriptionTier.NO_TIER
    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert world.stripe.subscriptions["sub_1"]["status"] == "past_due"
    assert world.stripe.invoices["in_1"]["status"] == "open"
    assert world.ledger.transactions == {}


@pytest.mark.asyncio
async def test_stripe_retry_paying_later_restores_access():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        assert world.tier == SubscriptionTier.NO_TIER

        world.stripe.card_pays("in_1")
        await reconcile_wallet_payment_on_paid_invoice(world.stripe.view("in_1"))
        await world.subscription_updated("sub_1")
        # The original failure, delivered again after the payment.
        await handle_subscription_payment_failure(event)

    assert world.tier == SubscriptionTier.PRO
    assert world.tier_writes == [SubscriptionTier.NO_TIER, SubscriptionTier.PRO]
    assert world.ledger.transactions == {}


@pytest.mark.asyncio
async def test_each_failed_retry_keeps_access_cut_without_cancelling():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        world.stripe.card_fails("in_1")
        await handle_subscription_payment_failure(event)

    assert world.tier == SubscriptionTier.NO_TIER
    assert world.tier_writes == [SubscriptionTier.NO_TIER]
    assert world.stripe.cancelled == []
    assert world.stripe.voided == []


@pytest.mark.asyncio
async def test_retry_after_a_top_up_is_paid_from_the_wallet():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        world.ledger.balance = 5000
        world.stripe.card_fails("in_1")
        await handle_subscription_payment_failure(event)
        await world.subscription_updated("sub_1")

    assert world.ledger.transactions == {"in_1": -2000}
    assert world.stripe.paid_out_of_band == ["in_1"]
    assert world.tier == SubscriptionTier.PRO


@pytest.mark.asyncio
async def test_old_failure_never_cancels_a_newer_active_subscription():
    """The reproduced defect: an old past-due subscription's failed retry
    used to cancel the customer's newer, paid subscription instead."""
    with World(balance=0) as world:
        event = _renewal_failed(world, sub_id="sub_old", invoice_id="in_old")
        world.stripe.add_subscription("sub_new", "active", latest_invoice="in_new")
        world.stripe.add_invoice("in_new", "sub_new", status="paid")
        await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.subscriptions["sub_new"]["status"] == "active"
    # The tier is recomputed from the failed subscription, so the sync sees
    # sub_new and keeps the paid tier, instead of a blanket NO_TIER write.
    assert [s["id"] for s in world.synced] == ["sub_old"]
    assert world.tier == SubscriptionTier.PRO
    assert world.tier_writes == []


def _superseded_by_a_newer_plan(world: World, balance: int) -> dict:
    """An old PRO subscription went past due, and the customer then bought
    MAX through Checkout, whose cleanup skips past-due subscriptions."""
    world.ledger.balance = balance
    world.tier = SubscriptionTier.MAX
    world.stripe.add_subscription(
        "sub_new", "active", latest_invoice="in_new", price="price_max"
    )
    world.stripe.add_invoice("in_new", "sub_new", status="paid")
    return _renewal_failed(world, sub_id="sub_old", invoice_id="in_old")


@pytest.mark.asyncio
async def test_wallet_never_pays_a_subscription_a_newer_plan_replaced():
    """Paying the old invoice would reactivate the old plan, and its update
    event would cancel the newer one as a stale duplicate."""
    with World() as world:
        event = _superseded_by_a_newer_plan(world, balance=5000)
        await handle_subscription_payment_failure(event)
        await world.subscription_updated("sub_old")

    assert world.ledger.transactions == {}
    assert world.stripe.paid_out_of_band == []
    assert world.stripe.subscriptions["sub_new"]["status"] == "active"
    assert "sub_new" not in world.stripe.cancelled
    assert world.tier == SubscriptionTier.MAX


@pytest.mark.asyncio
async def test_started_wallet_payment_is_refunded_once_a_newer_plan_replaced_it():
    with World(balance=5000) as world:
        event = _renewal_failed(world, sub_id="sub_old", invoice_id="in_old")
        world.stripe.fail_next["pay"] = 1
        with pytest.raises(stripe.APIConnectionError):
            await handle_subscription_payment_failure(event)
        world.stripe.add_subscription(
            "sub_new", "active", latest_invoice="in_new", price="price_max"
        )

        await handle_subscription_payment_failure(event)

    assert world.stripe.paid_out_of_band == []
    assert world.ledger.balance == 5000
    assert world.stripe.subscriptions["sub_new"]["status"] == "active"


@pytest.mark.asyncio
async def test_failure_of_an_invoice_that_is_no_longer_latest_does_nothing():
    with World(balance=5000) as world:
        world.stripe.add_subscription("sub_1", "past_due", latest_invoice="in_2")
        event = world.stripe.add_invoice("in_1", "sub_1")
        world.stripe.add_invoice("in_2", "sub_1")
        await handle_subscription_payment_failure(event)

    assert world.stripe.paid_out_of_band == []
    assert world.ledger.transactions == {}
    assert world.synced == []


@pytest.mark.asyncio
async def test_delayed_failure_after_the_customer_paid_does_nothing():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        world.stripe.card_pays("in_1")
        await handle_subscription_payment_failure(event)

    assert world.synced == []
    assert world.tier == SubscriptionTier.PRO


@pytest.mark.asyncio
async def test_duplicate_failure_event_writes_the_tier_once():
    with World(balance=0) as world:
        event = _renewal_failed(world)
        await handle_subscription_payment_failure(event)
        await handle_subscription_payment_failure(event)

    assert world.tier_writes == [SubscriptionTier.NO_TIER]
    assert world.stripe.cancelled == []


@pytest.mark.asyncio
async def test_failure_after_stripe_ended_the_subscription_does_nothing():
    """Stripe's own final action after its retries cancels the subscription;
    its deletion event syncs the tier, so a late failure has nothing to do."""
    with World(balance=5000) as world:
        event = _renewal_failed(world)
        world.stripe.subscriptions["sub_1"]["status"] = "canceled"
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {}
    assert world.synced == []


@pytest.mark.asyncio
async def test_payment_still_processing_leaves_the_wallet_alone():
    with World(balance=5000) as world:
        event = _renewal_failed(world, payment_intent="pi_1")
        world.stripe.payment_intents["pi_1"] = {"id": "pi_1", "status": "processing"}
        await handle_subscription_payment_failure(event)

    assert world.ledger.transactions == {}
    assert [s["status"] for s in world.synced] == ["past_due"]
    assert world.tier == SubscriptionTier.NO_TIER


@pytest.mark.asyncio
async def test_trial_first_invoice_takes_the_same_path():
    """An unconverted trial's first invoice stays open for a card fix like any
    other; the trial reconcile maps the past-due trial to NO_TIER."""
    with World(balance=0) as world:
        event = _renewal_failed(world)
        world.stripe.subscriptions["sub_1"]["metadata"] = {"trial_enrollment_id": "t1"}

        async def reconcile_trial(user_id: str, sub_id: str):
            sub = world.stripe.subscriptions[sub_id]
            assert sub["status"] == "past_due"
            await world._set_tier(user_id, SubscriptionTier.NO_TIER)
            return dict(sub), SubscriptionTier.NO_TIER

        with patch(
            "backend.data.credit.reconcile_trial_subscription",
            side_effect=reconcile_trial,
        ), patch("backend.data.credit.invalidate_subscription_caches"):
            await handle_subscription_payment_failure(event)

    assert world.stripe.cancelled == []
    assert world.stripe.voided == []
    assert world.stripe.invoices["in_1"]["status"] == "open"
    assert world.tier == SubscriptionTier.NO_TIER


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
    assert world.ledger.settled("in_1")
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
        world.stripe.card_pays("in_1")

        await reconcile_wallet_payment_on_paid_invoice(world.stripe.view("in_1"))
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
        await reconcile_wallet_payment_on_paid_invoice(world.stripe.view("in_1"))

    assert world.ledger.balance == 3000
    assert "in_1:wallet-refund" not in world.ledger.transactions


@pytest.mark.asyncio
async def test_non_subscription_invoice_is_ignored():
    with World(balance=5000) as world:
        await handle_subscription_payment_failure(
            {"id": "in_x", "customer": CUSTOMER, "amount_due": 2000}
        )

    assert world.ledger.transactions == {}
