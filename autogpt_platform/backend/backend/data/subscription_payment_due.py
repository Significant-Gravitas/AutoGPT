"""The open bill that keeps a failed renewal's customer off their plan.

A renewal whose charge failed leaves the subscription ``past_due`` (or
``unpaid``) with its invoice open, and Stripe keeps retrying it. Paying that
invoice through its hosted page restores the plan, and the billing period
then restarts from the payment (``subscription_billing_restart``). A new
Checkout would instead start a second plan, whose arrival cancels the old
one and voids its bill, so Stripe's retries stop. While such a bill is
payable, the paywall offers it and Checkout is refused.
"""

import asyncio
import logging

import stripe

from backend.data.credit import build_price_to_tier_map, replaced_by_newer_plan
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.stripe_invoice_payments import stripe_id
from backend.data.subscription_billing_restart_invoice import current_period_start
from backend.data.subscription_payment_due_cache import (
    PAYMENTS_DUE,
    SubscriptionPaymentDue,
)

logger = logging.getLogger(__name__)

_PAYMENT_DUE_STATUSES = ("past_due", "unpaid")
# Stripe marks the bill uncollectible once its retries run out, but its
# payment link still takes a payment.
_PAYABLE_INVOICE_STATUSES = ("open", "uncollectible")
UNPAID_BILL_BLOCKS_CHECKOUT = (
    "Your last payment didn't go through. Pay the open bill to restore your"
    " plan instead of starting a new one."
)


async def get_subscription_payment_due(
    customer_id: str | None, *, fresh: bool = False
) -> SubscriptionPaymentDue | None:
    """The customer's open renewal bill, newest plan first, if one is payable.

    Read through a short cache shared by every API worker, since the billing
    status asks on every page load; ``fresh`` reads Stripe directly, for a
    decision that changes the customer's billing."""
    if not customer_id:
        return None
    if fresh:
        return await _read_payment_due(customer_id)
    return await PAYMENTS_DUE.read(customer_id, lambda: _read_payment_due(customer_id))


async def _read_payment_due(customer_id: str) -> SubscriptionPaymentDue | None:
    # Stripe lists one status at a time; both lists are read together.
    pages = await asyncio.gather(
        *(
            stripe_call(
                stripe.Subscription.list_async,
                customer=customer_id,
                status=status,
                limit=10,
                expand=["data.latest_invoice"],
            )
            for status in _PAYMENT_DUE_STATUSES
        )
    )
    for subscriptions in pages:
        async for subscription in stripe_list_items(subscriptions):
            due = await _payment_due(customer_id, dict(subscription))
            if due is not None:
                return due
    return None


async def _payment_due(
    customer_id: str, subscription: dict
) -> SubscriptionPaymentDue | None:
    invoice = subscription.get("latest_invoice")
    if not isinstance(invoice, dict) or subscription.get("pending_update"):
        # A pending update's invoice holds a change, not the plan itself.
        return None
    if (
        invoice.get("status") not in _PAYABLE_INVOICE_STATUSES
        or stripe_id(invoice.get("customer")) != customer_id
        or (invoice.get("amount_due") or 0) <= 0
        or not invoice.get("currency")
        or not invoice.get("hosted_invoice_url")
    ):
        return None
    if await replaced_by_newer_plan(customer_id, subscription):
        return None
    items = (subscription.get("items") or {}).get("data") or []
    price_id = items[0].get("price", {}).get("id") if items else None
    tier = (await build_price_to_tier_map()).get(price_id or "")
    if tier is None:
        # A price no longer on sale: paying keeps the account on NO_TIER, so
        # the customer is offered the plans instead.
        return None
    return SubscriptionPaymentDue(
        amount_due=invoice["amount_due"],
        currency=invoice["currency"],
        payment_url=invoice["hosted_invoice_url"],
        plan=tier.value,
        due_since=current_period_start(subscription),
    )
