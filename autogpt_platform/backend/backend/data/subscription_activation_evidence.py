"""Validate settled invoices independently of cash collected (credits count)."""

import stripe
from pydantic import BaseModel, RootModel, TypeAdapter

from backend.data.stripe_client import stripe_call, stripe_list_items


class InvoiceId(RootModel[str]):
    @property
    def invoice_id(self) -> str:
        return self.root


class ExpandedInvoiceId(BaseModel):
    id: str

    @property
    def invoice_id(self) -> str:
        return self.id


INVOICE_REFERENCE = TypeAdapter(InvoiceId | ExpandedInvoiceId)


def qualifying_invoice(
    invoice: dict,
    *,
    customer_id: str,
    subscription_id: str,
    price_id: str,
    trial_end: int | None,
) -> bool:
    if not settled_recurring_invoice(invoice):
        return False
    if (
        invoice.get("customer") != customer_id
        or invoice_subscription(invoice) != subscription_id
    ):
        return False
    if trial_end is None:
        if invoice.get("billing_reason") != "subscription_create":
            return False
    elif (
        invoice.get("billing_reason")
        not in ("subscription_cycle", "subscription_update")
        or invoice.get("created", 0) < trial_end
    ):
        return False
    for line in invoice.get("lines", {}).get("data", []):
        period = line.get("period", {})
        if (
            _paid_recurring_line(line, subscription_id)
            and line_price(line) == price_id
            and line.get("quantity") == 1
            and not line.get("proration", False)
            and period.get("end", 0) > period.get("start", 0)
            and (trial_end is None or period.get("start") == trial_end)
        ):
            return True
    return False


def settled_recurring_invoice(invoice: dict) -> bool:
    return _settled_subscription_invoice(invoice) and any(
        _paid_recurring_line(line, invoice_subscription(invoice))
        for line in invoice.get("lines", {}).get("data", [])
    )


def settled_proration_invoice(invoice: dict, price_id: str | None) -> bool:
    """Current paid access evidence; never evidence of an initial usage reset."""
    return bool(
        price_id
        and invoice.get("billing_reason") == "subscription_update"
        and _settled_subscription_invoice(invoice)
        and any(
            _paid_recurring_line(
                line, invoice_subscription(invoice), allow_proration=True
            )
            and line_price(line) == price_id
            and line.get("quantity") == 1
            for line in invoice.get("lines", {}).get("data", [])
        )
    )


def _settled_subscription_invoice(invoice: dict) -> bool:
    return bool(
        invoice.get("status") == "paid"
        and invoice.get("amount_remaining") == 0
        and (invoice.get("status_transitions") or {}).get("paid_at")
        and invoice_subscription(invoice)
        and invoice.get("billing_reason")
        in ("subscription_create", "subscription_cycle", "subscription_update")
        and not invoice.get("lines", {}).get("has_more", False)
    )


def _paid_recurring_line(
    line: dict, subscription_id: str | None, *, allow_proration: bool = False
) -> bool:
    parent = line.get("parent") or {}
    details = parent.get("subscription_item_details") or {}
    recurring = (
        line.get("type") == "subscription"
        or parent.get("type") == "subscription_item_details"
    )
    owner = line.get("subscription") or details.get("subscription")
    period = line.get("period") or {}
    return bool(
        recurring
        and owner == subscription_id
        and subscription_id
        and (
            allow_proration
            or not line.get("proration", details.get("proration", False))
        )
        and period.get("end", 0) > period.get("start", 0)
        and _priced_service(line)
    )


def _priced_service(line: dict) -> bool:
    base = line.get("amount_excluding_tax") or line.get("amount", 0)
    discounts = sum(
        item.get("amount", 0) for item in line.get("discount_amounts") or []
    )
    return base > 0 or discounts > 0


def invoice_subscription(invoice: dict) -> str | None:
    return invoice.get("subscription") or (
        (invoice.get("parent") or {}).get("subscription_details") or {}
    ).get("subscription")


def line_price(line: dict) -> str | None:
    return (line.get("price") or {}).get("id") or (
        (line.get("pricing") or {}).get("price_details") or {}
    ).get("price")


async def first_settled_invoice(customer_id: str) -> dict | None:
    """Inspect all pages so returning subscribers never receive another reset."""
    invoices = await stripe_call(
        stripe.Invoice.list_async, customer=customer_id, status="paid", limit=100
    )
    first: dict | None = None
    async for raw in stripe_list_items(invoices):
        invoice = dict(raw)
        if not raw.id:
            raise ValueError("Stripe invoice has no durable identity")
        if invoice.get("lines", {}).get("has_more"):
            lines = await stripe_call(
                stripe.Invoice.list_lines_async, raw.id, limit=100
            )
            invoice["lines"] = {
                "data": [dict(line) async for line in stripe_list_items(lines)],
                "has_more": False,
            }
        if not settled_recurring_invoice(invoice):
            continue
        if first is None or _settled_order(invoice) < _settled_order(first):
            first = invoice
    return first


def _settled_order(invoice: dict) -> tuple[int, int, str]:
    return (invoice["status_transitions"]["paid_at"], invoice["created"], invoice["id"])
