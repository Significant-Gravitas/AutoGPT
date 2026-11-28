"""The invoice that charges a subscription's period, and the payment that pays it.

Stripe makes one when a subscription starts (``subscription_create``) and at
each renewal (``subscription_cycle``), and charges the subscription's card at
once. The platform grants a paid plan only on such a settled invoice, so these
carry what it checks: paid, nothing remaining, a paid-at time, and a line for
the subscription's price over the period it pays for.
"""

from typing import Any

from . import resources
from .store import StripeObject, new_id

INVOICE_PAGE = "/stripe/invoices"


def subscription_invoice(
    subscription: StripeObject,
    billing_reason: str,
    created: int,
    *,
    describe: dict[str, str],
    public_url: str,
) -> tuple[StripeObject, StripeObject]:
    """The paid invoice for the subscription's current period, and its payment.

    ``describe`` names each price (its product's name). The invoice has no
    number yet: ``Subscriptions.number_invoice`` gives it one as it is
    stored.
    """
    invoice_id = new_id("in")
    lines = [
        _line(item, subscription, invoice_id, describe)
        for item in subscription["items"]["data"]
    ]
    total = sum(line["amount"] for line in lines)
    payment = resources.invoice_payment(
        subscription, invoice_id, total, billing_reason, created
    )
    invoice = {
        "id": invoice_id,
        "object": "invoice",
        "amount_due": total,
        "amount_paid": total,
        "amount_remaining": 0,
        "attempt_count": 1,
        "attempted": True,
        "billing_reason": billing_reason,
        "collection_method": "charge_automatically",
        "created": created,
        "currency": subscription["currency"],
        "customer": subscription["customer"],
        "description": None,
        "hosted_invoice_url": f"{public_url}{INVOICE_PAGE}/{invoice_id}",
        "invoice_pdf": None,
        "lines": {
            "object": "list",
            "data": lines,
            "has_more": False,
            "total_count": len(lines),
            "url": f"/v1/invoices/{invoice_id}/lines",
        },
        "livemode": False,
        "metadata": {},
        "next_payment_attempt": None,
        "number": None,
        "paid": True,
        "paid_out_of_band": False,
        "payment_intent": payment["id"],
        "period_end": created,
        "period_start": created,
        "status": "paid",
        "status_transitions": {
            "finalized_at": created,
            "marked_uncollectible_at": None,
            "paid_at": created,
            "voided_at": None,
        },
        "subscription": subscription["id"],
        "subscription_details": {"metadata": subscription["metadata"]},
        "subtotal": total,
        "total": total,
    }
    return invoice, payment


def invoice_number(customer_id: str, sequence: int) -> str:
    """Stripe numbers a customer's invoices ``<prefix>-0001``, ``-0002``..."""
    prefix = customer_id.removeprefix("cus_")[:8].upper()
    return f"{prefix}-{sequence:04d}"


def _line(
    item: dict[str, Any],
    subscription: StripeObject,
    invoice_id: str,
    describe: dict[str, str],
) -> dict[str, Any]:
    price = item["price"]
    quantity = int(item["quantity"])
    amount = int(price["unit_amount"]) * quantity
    interval = price["recurring"]["interval"]
    name = describe.get(price["id"], price["id"])
    return {
        "id": new_id("il"),
        "object": "line_item",
        "amount": amount,
        "amount_excluding_tax": amount,
        "currency": price["currency"],
        "description": f"{quantity} × {name} (at {resources.money(int(price['unit_amount']), price['currency'])} / {interval})",
        "discount_amounts": [],
        "invoice": invoice_id,
        "livemode": False,
        "metadata": {},
        "period": {
            "start": subscription["current_period_start"],
            "end": subscription["current_period_end"],
        },
        "price": price,
        "proration": False,
        "quantity": quantity,
        "subscription": subscription["id"],
        "subscription_item": item["id"],
        "type": "subscription",
    }
