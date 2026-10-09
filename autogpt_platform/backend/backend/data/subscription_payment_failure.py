"""React to a failed subscription invoice from Stripe.

Stripe delivers ``invoice.payment_failed`` late, twice, and out of order, and
a customer can have an old failed subscription next to a newer paid one. So
the handler reads the invoice and its subscription fresh from Stripe, acts
only on that one subscription, and only while the invoice is still its
latest and still unpaid. Every step is safe to repeat, and any Stripe error
is raised so the webhook retries and resumes.
"""

import logging
from typing import Any

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User
from pydantic import BaseModel

from backend.data.credit import (
    PAYMENT_FAILURE_CANCELLATION_COMMENT,
    _cancel_params,
    _invoice_subscription_id,
    sync_subscription_from_stripe,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import get_subscription_trial
from backend.data.subscription_wallet_payment import (
    find_wallet_debit,
    pay_invoice_from_wallet,
    settle_wallet_payment,
)

logger = logging.getLogger(__name__)

# Subscription states in which a failed invoice means the customer stopped
# paying. ``incomplete`` is left alone: its first payment may still be
# authenticated, and Stripe expires it on its own.
_UNPAID_STATUSES = ("past_due", "unpaid")


class FailedInvoice(BaseModel):
    user_id: str
    customer_id: str
    invoice: dict[str, Any]
    subscription: dict[str, Any]

    @property
    def invoice_id(self) -> str:
        return self.invoice["id"]

    @property
    def sub_id(self) -> str:
        return self.subscription["id"]

    @property
    def is_latest(self) -> bool:
        return _stripe_id(self.subscription.get("latest_invoice")) == self.invoice_id

    @property
    def is_unpaid(self) -> bool:
        return (
            self.is_latest
            and self.invoice.get("status") == "open"
            and self.subscription.get("status") in _UNPAID_STATUSES
        )


async def handle_subscription_payment_failure(invoice: dict) -> None:
    """Pay the failed invoice from the wallet, or end that subscription.

    - A wallet payment already started for this invoice is finished first.
    - Balance covers it → debit the wallet and mark the invoice paid.
    - Otherwise → cancel that subscription, void its unpaid invoices so
      nothing more is collected for a plan the customer no longer has, and
      recompute the tier, which keeps any other active plan.
    - A first trial invoice stays open for card repair and a payment still
      processing is left to settle; the tier sync still cuts access.
    """
    failed = await _load_failed_invoice(invoice)
    if failed is None:
        return
    debit = await find_wallet_debit(failed.user_id, failed.invoice_id)
    if debit is not None:
        await settle_wallet_payment(
            failed.user_id, failed.invoice, debit, may_pay=failed.is_unpaid
        )
        return
    if failed.is_latest and failed.subscription.get("status") == "canceled":
        # An earlier delivery cancelled it but did not finish voiding.
        await _void_open_invoices(failed.sub_id)
        await sync_subscription_from_stripe(failed.subscription)
        return
    if not failed.is_unpaid:
        logger.info(
            f"Payment failure for invoice {failed.invoice_id} is stale: invoice"
            f" {failed.invoice.get('status')}, subscription {failed.sub_id}"
            f" {failed.subscription.get('status')}, latest={failed.is_latest}"
        )
        return
    if failed.invoice.get("amount_due", 0) <= 0:
        return
    if await pay_invoice_from_wallet(
        failed.user_id, failed.customer_id, failed.sub_id, failed.invoice
    ):
        return
    await _end_unpaid_subscription(failed)


async def _load_failed_invoice(invoice: dict) -> FailedInvoice | None:
    customer_id = invoice.get("customer")
    sub_id = _invoice_subscription_id(invoice)
    invoice_id = invoice.get("id") or ""
    if not customer_id or not sub_id or not invoice_id:
        return None
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if not user:
        logger.warning(f"Payment failure for unknown Stripe customer {customer_id}")
        return None
    if user.subscriptionTier == SubscriptionTier.ENTERPRISE:
        # Admin-managed; a self-service failure must not touch it.
        return None
    fresh_invoice = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
    subscription = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
    if (
        fresh_invoice.get("customer") != customer_id
        or subscription.get("customer") != customer_id
    ):
        logger.error(
            f"Invoice {invoice_id} or subscription {sub_id} does not belong to"
            f" customer {customer_id}; ignoring the payment failure"
        )
        return None
    return FailedInvoice(
        user_id=user.id,
        customer_id=customer_id,
        invoice=fresh_invoice,
        subscription=subscription,
    )


async def _end_unpaid_subscription(failed: FailedInvoice) -> None:
    if await _payment_in_progress(failed.invoice):
        logger.info(
            f"A payment on invoice {failed.invoice_id} is still processing;"
            f" cutting access but leaving subscription {failed.sub_id} to settle"
        )
        await sync_subscription_from_stripe(failed.subscription)
        return
    if await _is_unconverted_trial(failed.user_id, failed.sub_id):
        await sync_subscription_from_stripe(failed.subscription)
        return
    logger.info(
        f"Balance cannot cover invoice {failed.invoice_id} of user"
        f" {failed.user_id}; cancelling subscription {failed.sub_id}"
    )
    # Cancel first: Stripe stops retrying invoices of a cancelled subscription.
    # If voiding then fails, the retry finishes it from the cancelled state.
    canceled = await _cancel_subscription(failed.sub_id)
    await _void_open_invoices(failed.sub_id)
    await sync_subscription_from_stripe(canceled)


async def _cancel_subscription(sub_id: str) -> dict:
    try:
        canceled = await stripe_call(
            stripe.Subscription.cancel_async,
            sub_id,
            **_cancel_params(PAYMENT_FAILURE_CANCELLATION_COMMENT, trial=False),
        )
        return dict(canceled)
    except stripe.StripeError:
        current = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
        if current.get("status") == "canceled":
            return current
        raise


async def _void_open_invoices(sub_id: str) -> None:
    """Close a cancelled subscription's unpaid invoices.

    Cancelling only pauses automatic collection; the invoice stays payable
    through its payment link and the portal, which would take money for a plan
    the customer no longer has. An invoice with a payment still processing is
    left for that payment's own success or failure event.
    """
    invoices = await stripe_call(
        stripe.Invoice.list_async, subscription=sub_id, status="open", limit=100
    )
    async for invoice in stripe_list_items(invoices):
        invoice_id: str = invoice["id"]
        if await _payment_in_progress(dict(invoice)):
            logger.warning(f"Not voiding invoice {invoice_id}: payment processing")
            continue
        try:
            await stripe_call(stripe.Invoice.void_invoice_async, invoice_id)
        except stripe.StripeError:
            current = await stripe_call(stripe.Invoice.retrieve_async, invoice_id)
            if current.get("status") == "open":
                raise
            if current.get("status") == "paid":
                logger.error(
                    f"Invoice {invoice_id} was paid after subscription {sub_id}"
                    " was cancelled for non-payment; needs a manual fix"
                )


async def _payment_in_progress(invoice: dict) -> bool:
    """Whether a payment on the invoice is still settling, e.g. a bank debit."""
    payment_intent_id = _stripe_id(invoice.get("payment_intent"))
    if not payment_intent_id:
        return False
    payment_intent = await stripe_call(
        stripe.PaymentIntent.retrieve_async, payment_intent_id
    )
    return payment_intent.get("status") == "processing"


async def _is_unconverted_trial(user_id: str, sub_id: str) -> bool:
    """A trial's first invoice stays open so the customer can fix the card."""
    trial = await get_subscription_trial(user_id)
    return bool(
        trial and trial.converted_at is None and trial.subscription_id == sub_id
    )


def _stripe_id(value: Any) -> str:
    """The id of a Stripe reference, whether it arrived as an id or expanded."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("id") or ""
    return ""
