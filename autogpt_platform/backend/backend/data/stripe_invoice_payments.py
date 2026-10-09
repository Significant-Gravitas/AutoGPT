"""Read how an invoice was paid, whatever Stripe API version shaped it.

Stripe API 2025-03-31.basil removed ``paid``, ``paid_out_of_band``, ``charge``
and ``payment_intent`` from the Invoice and moved its payments to
``invoice.payments`` (InvoicePayment objects, returned only when expanded).
Webhook payloads use the endpoint's API version while our own calls use the
SDK's pinned one, so the same handler can see either shape and must never
read a missing field as "no".
"""

import logging
from typing import Any

import stripe

from backend.data.stripe_client import stripe_call

logger = logging.getLogger(__name__)

_BASIL = "2025-03-31"
# InvoicePayment types Stripe collected itself, as opposed to a payment
# recorded out of band.
_COLLECTED_PAYMENT_TYPES = ("payment_intent", "charge")


async def paid_by_stripe_collection(invoice: dict) -> bool | None:
    """Whether Stripe collected the paid invoice itself (card, bank debit)
    rather than it being marked paid out of band.

    None when the invoice does not say, which callers must treat as unknown.
    """
    if invoice.get("status") != "paid":
        return False
    if "paid_out_of_band" not in invoice and _payments(invoice) is None:
        invoice = await _retrieve_with_payments(invoice["id"])
    if "paid_out_of_band" in invoice:
        return invoice.get("status") == "paid" and not invoice["paid_out_of_band"]
    payments = _payments(invoice)
    if payments is None:
        return None
    return any(
        payment.get("status") == "paid"
        and _payment_source(payment).get("type") in _COLLECTED_PAYMENT_TYPES
        for payment in payments
    )


async def payment_in_progress(invoice: dict) -> bool:
    """Whether a payment on the invoice is still settling, e.g. a bank debit.

    When the invoice does not say, it answers True: the caller then only cuts
    access and leaves the subscription and invoice for Stripe to settle.
    """
    if "payment_intent" not in invoice and _payments(invoice) is None:
        invoice = await _retrieve_with_payments(invoice["id"])
    if "payment_intent" in invoice:
        payment_intent_ids = [stripe_id(invoice.get("payment_intent"))]
    elif (payments := _payments(invoice)) is not None:
        payment_intent_ids = [
            stripe_id(_payment_source(payment).get("payment_intent"))
            for payment in payments
            if payment.get("status") != "canceled"
        ]
    else:
        logger.error(
            f"Cannot tell whether a payment on invoice {invoice.get('id')} is"
            " processing; treating it as in progress"
        )
        return True
    for payment_intent_id in filter(None, payment_intent_ids):
        payment_intent = await stripe_call(
            stripe.PaymentIntent.retrieve_async, payment_intent_id
        )
        if payment_intent.get("status") == "processing":
            return True
    return False


def stripe_id(value: Any) -> str:
    """The id of a Stripe reference, whether it arrived as an id or expanded."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("id") or ""
    return ""


async def _retrieve_with_payments(invoice_id: str) -> dict:
    params = {"expand": ["payments"]} if stripe.api_version >= _BASIL else {}
    return dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id, **params))


def _payments(invoice: dict) -> list[dict] | None:
    """``invoice.payments`` when the invoice carries it (basil or later, expanded)."""
    payments = invoice.get("payments")
    return payments.get("data") if payments else None


def _payment_source(payment: dict) -> dict:
    return payment.get("payment") or {}
