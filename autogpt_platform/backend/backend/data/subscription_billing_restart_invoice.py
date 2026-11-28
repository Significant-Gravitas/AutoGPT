"""What a billing restart reads off Stripe invoices and subscriptions.

Stripe API 2025-03-31.basil moved the current period from the subscription
to its items, and a line's subscription details from the line to its
``parent``; both shapes are read here.
"""

import logging

import stripe

from backend.data.stripe_client import stripe_call

logger = logging.getLogger(__name__)


def current_period_start(subscription: dict) -> int | None:
    """The current period's start, on the subscription before Stripe API
    2025-03-31.basil and on its items from then on."""
    if subscription.get("current_period_start"):
        return subscription["current_period_start"]
    items = (subscription.get("items") or {}).get("data") or []
    starts = [
        item["current_period_start"]
        for item in items
        if item.get("current_period_start")
    ]
    return min(starts) if starts else None


def line_period_start(invoice: dict) -> int | None:
    """When the period the invoice's recurring plan line bills began."""
    starts = [
        line["period"]["start"]
        for line in _lines(invoice)
        if _is_recurring_line(line) and (line.get("period") or {}).get("start")
    ]
    return min(starts) if starts else None


def bills_only_the_plan(invoice: dict) -> bool:
    """Whether every line of the invoice is the plan's recurring charge.

    The reset invoice bills only the plan, so crediting a one-off line too
    would leave it on the customer's balance to pay a later invoice. A page
    of lines that is not the whole invoice cannot be judged either.
    """
    lines = invoice.get("lines") or {}
    data = lines.get("data") or []
    return (
        bool(data) and not lines.get("has_more") and all(map(_is_recurring_line, data))
    )


async def credit_covers_new_period(
    sub_id: str, invoice_id: str, credit: int, *, credited: bool
) -> bool:
    """Whether the late invoice's credit pays the whole reset invoice.

    The reset bills the plan at today's terms, after the customer's balance.
    When the late invoice cost less (a once-only promotion, a price change
    since) or the customer owes a balance, the card would be charged the
    difference without the customer agreeing to it, so the restart is left
    for a person instead. ``credited`` says the credit is already on the
    balance the preview reads.
    """
    preview = await stripe_call(
        stripe.Invoice.create_preview_async,
        subscription=sub_id,
        subscription_details={
            "billing_cycle_anchor": "now",
            "proration_behavior": "none",
        },
    )
    new_period: int = preview.get("total") or 0
    amount_due: int = preview.get("amount_due") or 0
    left_for_card = amount_due if credited else amount_due - credit
    if new_period <= credit and left_for_card <= 0:
        return True
    logger.error(
        f"Restarting subscription {sub_id} from late invoice {invoice_id}"
        f" would cost {new_period} against a {credit} credit, leaving"
        f" {max(0, left_for_card)} for the card; the billing period was not"
        " restarted"
    )
    return False


def _lines(invoice: dict) -> list[dict]:
    return (invoice.get("lines") or {}).get("data") or []


def _is_recurring_line(line: dict) -> bool:
    parent = line.get("parent") or {}
    if parent.get("type") == "subscription_item_details":
        details = parent.get("subscription_item_details") or {}
        return not details.get("proration")
    return line.get("type") == "subscription" and not line.get("proration")
