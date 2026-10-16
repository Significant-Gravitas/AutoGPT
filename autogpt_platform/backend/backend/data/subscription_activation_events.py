"""A crashed webhook claim must not suppress authoritative activation recovery."""

import stripe

from backend.data import credit
from backend.data.stripe_client import stripe_call
from backend.data.subscription_activation_evidence import invoice_subscription


async def recover_claimed_billing_event(event_type: str, payload: dict) -> None:
    """Only reconcile existing objects; never create checkout or retry collection."""
    if event_type.startswith("customer.subscription."):
        await credit.sync_subscription_from_stripe(payload)
        return
    if event_type in (
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
    ):
        await credit.sync_tier_from_checkout_session(payload)
        return
    if event_type in ("invoice_payment.paid", "invoice_payment.payment_failed"):
        invoice_id = payload.get("invoice")
        if not invoice_id:
            return
        payload = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
    elif event_type not in (
        "invoice.paid",
        "invoice.payment_succeeded",
        "invoice.payment_failed",
    ):
        return
    subscription_id = invoice_subscription(payload)
    if subscription_id:
        subscription = await stripe_call(
            stripe.Subscription.retrieve_async, subscription_id
        )
        await credit.sync_subscription_from_stripe(dict(subscription))
