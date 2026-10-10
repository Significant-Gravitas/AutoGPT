"""React to a failed subscription invoice from Stripe.

Stripe delivers ``invoice.payment_failed`` late, twice, and out of order, and
a customer can have an old failed subscription next to a newer paid one. So
the handler reads the invoice and its subscription fresh from Stripe, acts
only on that one subscription, and only while the invoice is still its
latest and still unpaid. Every step is safe to repeat, and any Stripe error
is raised so the webhook retries and resumes.

An unpaid subscription is not cancelled: access ends at once, while Stripe
keeps retrying the invoice and its payment link stays usable. Paying it
later restores access through the subscription's own update event.
"""

import logging
from typing import Any

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User
from pydantic import BaseModel

from backend.data.credit import _invoice_subscription_id, sync_subscription_from_stripe
from backend.data.stripe_client import stripe_call
from backend.data.stripe_invoice_payments import payment_in_progress, stripe_id
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
        return stripe_id(self.subscription.get("latest_invoice")) == self.invoice_id

    @property
    def is_unpaid(self) -> bool:
        return (
            self.is_latest
            and self.invoice.get("status") == "open"
            and self.subscription.get("status") in _UNPAID_STATUSES
        )


async def handle_subscription_payment_failure(invoice: dict) -> None:
    """Pay the failed invoice from the wallet, or cut access until it is paid.

    - A payment still processing (e.g. a bank debit) is left to settle before
      anything else: the wallet is not touched and the tier sync cuts access.
    - A wallet payment already started for this invoice is finished first.
    - Balance covers it → debit the wallet and mark the invoice paid.
    - Otherwise → recompute the tier from that subscription, which is
      ``past_due`` or ``unpaid`` and so gives no access unless another plan
      is active. Nothing changes in Stripe: its retries and the payment link
      stay open, and a trial's first invoice takes the same path.
    """
    failed = await _load_failed_invoice(invoice)
    if failed is None:
        return
    payment = await find_wallet_payment(failed.user_id, failed.invoice_id)
    if await _left_to_settle(failed, payment):
        return
    if payment is not None:
        await settle_wallet_payment(
            failed.user_id, failed.invoice_id, may_pay=failed.is_unpaid
        )
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
    logger.info(
        f"Balance cannot cover invoice {failed.invoice_id} of user"
        f" {failed.user_id}; cutting access while Stripe retries subscription"
        f" {failed.sub_id}"
    )
    await sync_subscription_from_stripe(failed.subscription)


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
