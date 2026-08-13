"""Turning a Stripe subscription payload into the plan slots the billing
emails are written from.

Every sentence in those emails is assembled from these values — "this month's
renewal" vs "this year's renewal", the plan name, the amount — so getting them
from Stripe rather than from our own copy is what keeps the copy honest.
"""

import logging
from datetime import datetime, timezone
from typing import Literal

from backend.data.credit import build_price_to_tier_map
from backend.data.notifications import CardDetails, SubscriptionPlan
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Lifecycle]")

_ZERO_DECIMAL_CURRENCIES = {
    "bif",
    "clp",
    "djf",
    "gnf",
    "jpy",
    "kmf",
    "krw",
    "mga",
    "pyg",
    "rwf",
    "ugx",
    "vnd",
    "vuv",
    "xaf",
    "xof",
    "xpf",
}


async def plan_from_subscription(subscription: dict) -> SubscriptionPlan:
    """Read the plan the customer is actually on out of a Stripe subscription."""
    price = _first_price(subscription)
    cycle = _cycle(price)
    name = await _tier_name(price.get("id"))
    return SubscriptionPlan(
        name=name,
        cycle=cycle,
        cycle_noun="month" if cycle == "monthly" else "year",
        label=f"{name} — {cycle}",
        price_display=f"{format_amount(price.get('unit_amount'), price.get('currency', 'usd'))} / {'month' if cycle == 'monthly' else 'year'}",
    )


async def plan_from_invoice(invoice: dict) -> SubscriptionPlan:
    """Same, from the invoice shape the payment events carry."""
    lines = (invoice.get("lines") or {}).get("data") or []
    price = (lines[0].get("price") or {}) if lines else {}
    cycle = _cycle(price)
    name = await _tier_name(price.get("id"))
    return SubscriptionPlan(
        name=name,
        cycle=cycle,
        cycle_noun="month" if cycle == "monthly" else "year",
        label=f"{name} — {cycle}",
        price_display=f"{format_amount(price.get('unit_amount'), price.get('currency', 'usd'))} / {'month' if cycle == 'monthly' else 'year'}",
    )


def card_from_invoice(invoice: dict) -> CardDetails:
    """Card brand and last four for the "Card ···· 4242" row. Falls back to
    neutral wording rather than inventing digits."""
    payment = (
        (invoice.get("payment_intent") or {}).get("charges", {}).get("data") or [{}]
    )[0]
    details = ((payment.get("payment_method_details") or {}).get("card")) or {}
    if not details:
        details = ((invoice.get("default_payment_method") or {}).get("card")) or {}
    return CardDetails(
        brand=str(details.get("brand") or "Card").title(),
        last4=str(details.get("last4") or "••••"),
    )


def format_amount(minor_units: int | None, currency: str) -> str:
    """Stripe amounts are in the currency's minor unit, except where they
    aren't."""
    if minor_units is None:
        return "—"
    symbol = {"usd": "$", "eur": "€", "gbp": "£"}.get(currency.lower(), "")
    if currency.lower() in _ZERO_DECIMAL_CURRENCIES:
        return f"{symbol}{minor_units:,}"
    return f"{symbol}{minor_units / 100:,.2f}"


def format_date(timestamp: int | None) -> str:
    """The one fact these emails exist to state, so it is never relative."""
    if not timestamp:
        return "—"
    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return f"{moment.day} {moment.strftime('%b %Y')}"


def _first_price(subscription: dict) -> dict:
    items = (subscription.get("items") or {}).get("data") or []
    return (items[0].get("price") or {}) if items else {}


def _cycle(price: dict) -> Literal["monthly", "yearly"]:
    interval = ((price.get("recurring") or {}).get("interval")) or "month"
    return "yearly" if interval == "year" else "monthly"


async def _tier_name(price_id: str | None) -> str:
    if not price_id:
        return "AutoGPT"
    try:
        tier = (await build_price_to_tier_map()).get(price_id)
    except Exception:
        logger.warning(f"Could not resolve tier for price {price_id}", exc_info=True)
        return "AutoGPT"
    return tier.value.title() if tier else "AutoGPT"
