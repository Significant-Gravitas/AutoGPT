"""The Stripe objects the fake makes, in the shape the API returns them.

Only the fields the platform reads, plus the ones that identify an object to
the SDK (``id``, ``object``), are filled; the rest Stripe would send are left
out rather than invented.
"""

from decimal import ROUND_CEILING, Decimal
from typing import Any

from .forms import as_int
from .store import StripeObject, new_id, now

# A session's ``url`` and a portal session's are pages on the fake itself.
CHECKOUT_PAGE = "/stripe/checkout"
PORTAL_PAGE = "/stripe/billing_portal"


def customer(params: dict[str, Any]) -> StripeObject:
    return {
        "id": new_id("cus"),
        "object": "customer",
        "address": params.get("address"),
        "balance": 0,
        "created": now(),
        "default_source": None,
        "delinquent": False,
        "email": params.get("email"),
        "invoice_settings": {"default_payment_method": None},
        "livemode": False,
        "metadata": params.get("metadata") or {},
        "name": params.get("name"),
    }


def line_item(
    description: str,
    unit_price: Decimal,
    quantity: Any,
    currency: str,
    price: StripeObject | None = None,
) -> dict[str, Any]:
    """As Stripe totals a line: the unit price times the quantity, rounded up
    to a whole minor unit (0.05 x 30 is 1.5, charged as 2). ``price`` is the
    price it was bought at, when it names one."""
    count = as_int(quantity, 1)
    total = (unit_price * count).to_integral_value(rounding=ROUND_CEILING)
    recurring = (price or {}).get("recurring") or {}
    return {
        "description": description,
        "quantity": count,
        "amount_total": int(total),
        "currency": currency,
        "price": price["id"] if price else None,
        "interval": recurring.get("interval"),
    }


def subscription(
    session: StripeObject,
    prices: list[tuple[StripeObject, int]],
    payment_method_id: str,
    start: int,
    end: int,
) -> StripeObject:
    """The active subscription a paid checkout starts: one item per recurring
    price, charged to the card the checkout took."""
    subscription_id = new_id("sub")
    items = [
        {
            "id": new_id("si"),
            "object": "subscription_item",
            "created": start,
            "metadata": {},
            "price": price,
            "quantity": quantity,
            "subscription": subscription_id,
        }
        for price, quantity in prices
    ]
    return {
        "id": subscription_id,
        "object": "subscription",
        "billing_cycle_anchor": start,
        "cancel_at": None,
        "cancel_at_period_end": False,
        "canceled_at": None,
        "cancellation_details": not_cancelled(),
        "collection_method": "charge_automatically",
        "created": start,
        "currency": prices[0][0]["currency"],
        "current_period_end": end,
        "current_period_start": start,
        "customer": session["customer"],
        "default_payment_method": payment_method_id,
        "discounts": [],
        "ended_at": None,
        "items": {
            "object": "list",
            "data": items,
            "has_more": False,
            "total_count": len(items),
            "url": f"/v1/subscription_items?subscription={subscription_id}",
        },
        "latest_invoice": None,
        "livemode": False,
        "metadata": dict(session["_subscription_data"].get("metadata") or {}),
        "schedule": None,
        "start_date": start,
        "status": "active",
        "trial_end": None,
        "trial_start": None,
    }


def not_cancelled() -> dict[str, Any]:
    return {"comment": None, "feedback": None, "reason": None}


def invoice_payment(
    subscription: StripeObject,
    invoice_id: str,
    amount: int,
    billing_reason: str,
    created: int,
) -> StripeObject:
    """The payment that settles a subscription invoice with its card."""
    description = (
        "Subscription creation"
        if billing_reason == "subscription_create"
        else "Subscription update"
    )
    return {
        "id": new_id("pi"),
        "object": "payment_intent",
        "amount": amount,
        "amount_received": amount,
        "created": created,
        "currency": subscription["currency"],
        "customer": subscription["customer"],
        "description": description,
        "invoice": invoice_id,
        "latest_charge": new_id("ch"),
        "livemode": False,
        "metadata": {},
        "payment_method": subscription["default_payment_method"],
        "setup_future_usage": "off_session",
        "status": "succeeded",
    }


def money(amount: int, currency: str | None) -> str:
    """An amount in a currency's minor unit, as a page shows it: ``$23.00``."""
    if (currency or "usd").lower() == "usd":
        return f"${amount / 100:,.2f}"
    return f"{amount / 100:,.2f} {str(currency).upper()}"


def checkout_session(
    params: dict[str, Any], items: list[dict[str, Any]], public_url: str
) -> StripeObject:
    session_id = new_id("cs_test")
    created = now()
    total = sum(item["amount_total"] for item in items)
    return {
        "id": session_id,
        "object": "checkout.session",
        "amount_subtotal": total,
        "amount_total": total,
        "cancel_url": params.get("cancel_url"),
        "client_reference_id": params.get("client_reference_id"),
        "created": created,
        "currency": items[0]["currency"] if items else None,
        "customer": params.get("customer"),
        "customer_details": None,
        "customer_email": params.get("customer_email"),
        "expires_at": created + 24 * 3600,
        "invoice": None,
        "livemode": False,
        "metadata": params.get("metadata") or {},
        "mode": params.get("mode"),
        "payment_intent": None,
        "payment_status": "unpaid",
        "status": "open",
        "subscription": None,
        "success_url": params.get("success_url"),
        "ui_mode": params.get("ui_mode") or "hosted",
        "url": f"{public_url}{CHECKOUT_PAGE}/{session_id}",
        "_line_items": items,
        "_payment_intent_data": params.get("payment_intent_data") or {},
        "_subscription_data": params.get("subscription_data") or {},
    }


def card(customer_id: str | None, allow_redisplay: str = "always") -> StripeObject:
    """Stripe's 4242 test Visa; ``customer`` is set when it was saved. A card
    a subscription checkout saves is ``limited``: shown again only where a
    checkout asks for limited cards too."""
    return {
        "id": new_id("pm"),
        "object": "payment_method",
        "allow_redisplay": allow_redisplay,
        "billing_details": {"address": {"country": "US"}, "email": None, "name": None},
        "card": {
            "brand": "visa",
            "country": "US",
            "exp_month": 12,
            "exp_year": 2034,
            "funding": "credit",
            "last4": "4242",
        },
        "created": now(),
        "customer": customer_id,
        "livemode": False,
        "metadata": {},
        "type": "card",
    }


def payment_intent(session: StripeObject, payment_method_id: str) -> StripeObject:
    data = session["_payment_intent_data"]
    return {
        "id": new_id("pi"),
        "object": "payment_intent",
        "amount": session["amount_total"],
        "amount_received": session["amount_total"],
        "created": now(),
        "currency": session["currency"],
        "customer": session["customer"],
        "description": data.get("description"),
        "latest_charge": new_id("ch"),
        "livemode": False,
        "metadata": data.get("metadata") or {},
        "payment_method": payment_method_id,
        "setup_future_usage": data.get("setup_future_usage"),
        "status": "succeeded",
    }


def customer_details(owner: StripeObject | None) -> dict[str, Any]:
    """What a buyer types on the checkout page: a US billing address."""
    return {
        "address": {"country": "US", "postal_code": "94103"},
        "email": owner.get("email") if owner else None,
        "name": owner.get("name") if owner else None,
        "tax_exempt": "none",
    }


def portal_session(params: dict[str, Any], public_url: str) -> StripeObject:
    session_id = new_id("bps")
    return {
        "id": session_id,
        "object": "billing_portal.session",
        "created": now(),
        "customer": params.get("customer"),
        "livemode": False,
        "return_url": params.get("return_url"),
        "url": f"{public_url}{PORTAL_PAGE}/{session_id}",
    }
