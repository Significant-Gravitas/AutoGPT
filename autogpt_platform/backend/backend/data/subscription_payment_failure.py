"""React to a failed subscription invoice from Stripe.

Stripe delivers ``invoice.payment_failed`` late, twice, and out of order, and
a customer can have an old failed subscription next to a newer paid one. So
the handler reads the invoice and its subscription fresh from Stripe, acts
only on that one subscription, and only while the invoice is still its
latest and still unpaid. Every step is safe to repeat, and any Stripe error
is raised so the webhook retries and resumes.

An unpaid subscription that is the customer's only plan is not cancelled:
access ends at once, while Stripe keeps retrying the invoice and its payment
link stays usable. Paying it later restores access through the
subscription's own update event. One the customer has replaced with a newer
active or trialing plan is cancelled and its unpaid invoices voided instead:
paying it would reactivate the old plan, whose update event would then
cancel the newer one as a duplicate.
"""

import logging
from enum import StrEnum
from typing import Any

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User
from pydantic import BaseModel

from backend.data.credit import (
    REPLACED_PLAN_CANCELLATION_COMMENT,
    _invoice_subscription_id,
    replaced_by_newer_plan,
    sync_subscription_from_stripe,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.stripe_invoice_payments import payment_in_progress, stripe_id
from backend.data.subscription_trial import get_subscription_trial
from backend.data.subscription_wallet_payment import (
    pay_invoice_from_wallet,
    settle_wallet_payment,
)
from backend.data.wallet_payment_state import (
    WalletPayment,
    WalletPaymentState,
    find_wallet_payment,
)

logger = logging.getLogger(__name__)

# Subscription states in which a failed invoice means the customer stopped
# paying. ``incomplete`` is left alone: its first payment may still be
# authenticated, and Stripe expires it on its own.
_UNPAID_STATUSES = ("past_due", "unpaid")
# Invoice states still payable through a payment link or the portal.
_PAYABLE_INVOICE_STATUSES = ("open", "uncollectible")
# Tiers with no paid plan to keep: the wallet starts no payment for them.
_UNPAID_TIERS = (SubscriptionTier.NO_TIER, SubscriptionTier.TRIAL)


class FailedInvoice(BaseModel):
    user_id: str
    user_tier: SubscriptionTier
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
        return stripe_id(self.subscription.get("latest_invoice")) == self.invoice_id

    @property
    def was_replaced(self) -> bool:
        """Cancelled by us because another plan replaced it."""
        details = self.subscription.get("cancellation_details") or {}
        return (
            self.subscription.get("status") == "canceled"
            and details.get("comment") == REPLACED_PLAN_CANCELLATION_COMMENT
        )

    @property
    def waits_on_pending_update(self) -> bool:
        """Stripe applies a pending update only once its invoice is paid, and
        voids that invoice when the update lapses, so a failed payment on it
        changed nothing. A trial converting early waits like this for the
        customer's bank to confirm."""
        return self.is_latest and bool(self.subscription.get("pending_update"))

    @property
    def is_unpaid(self) -> bool:
        return (
            self.is_latest
            and not self.waits_on_pending_update
            and self.invoice.get("status") == "open"
            and self.subscription.get("status") in _UNPAID_STATUSES
        )


class PaymentFailureOutcome(StrEnum):
    NEEDS_ACTION = "needs_action"
    SETTLED = "settled"
    IGNORED = "ignored"


async def handle_subscription_payment_failure(invoice: dict) -> PaymentFailureOutcome:
    """Pay the failed invoice from the wallet, or cut access until it is paid.

    - A payment still processing (e.g. a bank debit) is left to settle before
      anything else: the wallet is not touched and the tier sync cuts access.
    - A subscription a newer active or trialing plan replaced is cancelled
      and its unpaid invoices voided, never paid: paying would reactivate
      it, and its update event would then cancel the newer plan as a
      duplicate. A wallet debit already started for it is refunded.
    - An invoice a pending update waits on, or one Stripe has since paid or
      voided, covers nothing: the change it holds applies only once paid.
    - A wallet payment already started for this invoice is finished first.
    - Balance covers it → debit the wallet and mark the invoice paid, unless
      the user has no paid tier left (``NO_TIER`` or ``TRIAL``) or it is an
      unconverted trial's own invoice, which waits for the card instead.
    - Otherwise → recompute the tier from that subscription, which is
      ``past_due`` or ``unpaid`` and so gives no access unless another plan
      is active. Nothing changes in Stripe: its retries and the payment link
      stay open, and a trial's first invoice takes the same path.
    """
    failed = await _load_failed_invoice(invoice)
    if failed is None:
        return PaymentFailureOutcome.IGNORED
    payment = await find_wallet_payment(failed.user_id, failed.invoice_id)
    if await _left_to_settle(failed, payment):
        return PaymentFailureOutcome.IGNORED
    replaced = failed.is_unpaid and await replaced_by_newer_plan(
        failed.customer_id, failed.subscription
    )
    if payment is not None:
        await settle_wallet_payment(
            failed.user_id, failed.invoice_id, may_pay=failed.is_unpaid and not replaced
        )
        # Settling may have just refunded it.
        payment = (
            await find_wallet_payment(failed.user_id, failed.invoice_id) or payment
        )
    if replaced:
        await _end_replaced_subscription(failed)
        return PaymentFailureOutcome.IGNORED
    if failed.was_replaced:
        # An earlier delivery cancelled it but did not finish voiding. Any of
        # its invoices' events may finish it, not only the latest one's.
        await _void_unpaid_invoices(failed.sub_id)
        return PaymentFailureOutcome.IGNORED
    if payment is not None:
        # A refunded debit never pays the invoice again, so the customer must.
        if payment.state == WalletPaymentState.REFUNDED and (
            failed.is_unpaid or await _uncollectible_awaits_customer(failed)
        ):
            return PaymentFailureOutcome.NEEDS_ACTION
        return PaymentFailureOutcome.IGNORED
    if not failed.is_unpaid:
        if await _uncollectible_awaits_customer(failed):
            return PaymentFailureOutcome.NEEDS_ACTION
        logger.info(
            f"Payment failure for invoice {failed.invoice_id} is stale: invoice"
            f" {failed.invoice.get('status')}, subscription {failed.sub_id}"
            f" {failed.subscription.get('status')}, latest={failed.is_latest},"
            f" pending update={failed.waits_on_pending_update}"
        )
        return PaymentFailureOutcome.IGNORED
    if failed.invoice.get("amount_due", 0) <= 0:
        return PaymentFailureOutcome.IGNORED
    if await _wallet_may_start_payment(failed) and await pay_invoice_from_wallet(
        failed.user_id, failed.customer_id, failed.sub_id, failed.invoice
    ):
        return PaymentFailureOutcome.SETTLED
    logger.info(
        f"Wallet did not pay invoice {failed.invoice_id} of user"
        f" {failed.user_id}; cutting access while Stripe retries subscription"
        f" {failed.sub_id}"
    )
    await sync_subscription_from_stripe(failed.subscription)

    return PaymentFailureOutcome.NEEDS_ACTION


async def _left_to_settle(failed: FailedInvoice, payment: WalletPayment | None) -> bool:
    """Leave the invoice alone while a payment on it is still processing.

    Stripe will not mark the invoice paid while it settles, so a wallet
    payment would only fail and retry; that payment's own success or failure
    event decides what happens next. Access is still cut by the tier sync.
    """
    wallet_unfinished = (
        payment is not None and payment.state == WalletPaymentState.DEBITED
    )
    if not (failed.is_unpaid or wallet_unfinished):
        return False
    if not await payment_in_progress(failed.invoice):
        return False
    logger.info(
        f"A payment on invoice {failed.invoice_id} is still processing;"
        f" cutting access but leaving subscription {failed.sub_id} to settle"
    )
    await sync_subscription_from_stripe(failed.subscription)
    return True


async def _uncollectible_awaits_customer(failed: FailedInvoice) -> bool:
    """Stripe can mark the invoice uncollectible once its retries run out. It
    stays payable through its link, so the customer is still told, unless a
    newer plan replaced this one: paying would reactivate it. The wallet
    never pays it here; this only decides the notice."""
    if not (
        failed.is_latest
        and not failed.waits_on_pending_update
        and failed.invoice.get("status") == "uncollectible"
        and failed.invoice.get("amount_due", 0) > 0
        and failed.subscription.get("status") in _UNPAID_STATUSES
    ):
        return False
    return not await replaced_by_newer_plan(failed.customer_id, failed.subscription)


async def _wallet_may_start_payment(failed: FailedInvoice) -> bool:
    """An unconverted trial's first invoice stays open for authentication or
    card repair, and a user without a paid tier has no plan to keep, so
    neither spends wallet credit. A payment already started is still
    finished by ``settle_wallet_payment``."""
    if failed.user_tier in _UNPAID_TIERS:
        return False
    trial = await get_subscription_trial(failed.user_id)
    return not (
        trial is not None
        and trial.converted_at is None
        and trial.subscription_id == failed.sub_id
    )


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
        user_tier=user.subscriptionTier or SubscriptionTier.NO_TIER,
        customer_id=customer_id,
        invoice=fresh_invoice,
        subscription=subscription,
    )


async def _end_replaced_subscription(failed: FailedInvoice) -> None:
    logger.info(
        f"Subscription {failed.sub_id} of user {failed.user_id} was replaced by"
        f" another plan; cancelling it and voiding invoice {failed.invoice_id}"
    )
    # Cancel first: Stripe stops retrying invoices of a cancelled subscription.
    # If voiding then fails, the retry finishes it from the cancelled state.
    canceled = await _cancel_subscription(failed.sub_id)
    await _void_unpaid_invoices(failed.sub_id)
    await sync_subscription_from_stripe(canceled)


async def end_revived_plan(sub_id: str) -> None:
    """End an old plan that was paid after a newer plan replaced it.

    Stripe's card retry or the payment link can pay an old ``past_due`` plan
    before any failure event reaches us. It is ended like a replaced plan, so
    its update event never cancels the newer one. The payment that revived it
    stays with Stripe and needs a manual refund.
    """
    logger.error(
        f"Subscription {sub_id} was paid after a newer plan replaced it;"
        " cancelling it. Its last payment needs a manual refund"
    )
    await _cancel_subscription(sub_id)
    await _void_unpaid_invoices(sub_id)


async def _cancel_subscription(sub_id: str) -> dict:
    try:
        canceled = await stripe_call(
            stripe.Subscription.cancel_async,
            sub_id,
            cancellation_details={"comment": REPLACED_PLAN_CANCELLATION_COMMENT},
        )
        return dict(canceled)
    except stripe.StripeError:
        current = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
        if current.get("status") == "canceled":
            return current
        raise


async def _void_unpaid_invoices(sub_id: str) -> None:
    """Close a replaced subscription's unpaid invoices.

    Cancelling only pauses automatic collection; an ``open`` or
    ``uncollectible`` invoice stays payable through its payment link and the
    portal, which would charge for a plan the customer has replaced. An
    invoice with a payment still processing is left for that payment's own
    success or failure event.
    """
    for status in _PAYABLE_INVOICE_STATUSES:
        invoices = await stripe_call(
            stripe.Invoice.list_async, subscription=sub_id, status=status, limit=100
        )
        async for invoice in stripe_list_items(invoices):
            await _void_invoice(sub_id, dict(invoice))


async def _void_invoice(sub_id: str, invoice: dict) -> None:
    invoice_id: str = invoice["id"]
    if await payment_in_progress(invoice):
        logger.warning(f"Not voiding invoice {invoice_id}: payment processing")
        return
    try:
        await stripe_call(stripe.Invoice.void_invoice_async, invoice_id)
    except stripe.StripeError:
        current = await stripe_call(stripe.Invoice.retrieve_async, invoice_id)
        if current.get("status") in _PAYABLE_INVOICE_STATUSES:
            raise
        if current.get("status") == "paid":
            logger.error(
                f"Invoice {invoice_id} was paid after subscription {sub_id}"
                " was cancelled as replaced; needs a manual fix"
            )
