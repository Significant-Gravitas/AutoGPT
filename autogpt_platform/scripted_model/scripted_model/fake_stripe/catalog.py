"""The products and prices the fake Stripe sells, from ``catalog.json``.

They are seeded when the fake starts, under fixed ids: the platform reads its
plans' price ids from a flag (``copilot-tier-stripe-prices``), which the
journeys stack builds from the same file. Each price's ``lookup_key`` is the
key that flag gives it (``PRO``, ``PRO_YEARLY``).
"""

import json
from pathlib import Path
from typing import Any

from .store import Store, StripeObject, now

CATALOG_FILE = Path(__file__).with_name("catalog.json")


def seed(store: Store) -> None:
    """Adds every product and price in the catalogue to ``store``."""
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    created = now()
    for entry in catalog["products"]:
        store.add(_product(entry, created))
    for entry in catalog["prices"]:
        store.add(_price(entry, created))


def product_name(store: Store, price: StripeObject) -> str:
    """What a buyer sees a price called: its product's name, as Stripe shows
    it on checkout pages and invoices."""
    product = store.get(str(price.get("product")), "product")
    if product is not None:
        return str(product["name"])
    return str(price.get("nickname") or price["id"])


def _product(entry: dict[str, Any], created: int) -> StripeObject:
    return {
        "id": entry["id"],
        "object": "product",
        "active": True,
        "created": created,
        "livemode": False,
        "metadata": {},
        "name": entry["name"],
    }


def _price(entry: dict[str, Any], created: int) -> StripeObject:
    return {
        "id": entry["id"],
        "object": "price",
        "active": True,
        "billing_scheme": "per_unit",
        "created": created,
        "currency": "usd",
        "livemode": False,
        "lookup_key": entry["lookup_key"],
        "metadata": {},
        "nickname": None,
        "product": entry["product"],
        "recurring": {
            "interval": entry["interval"],
            "interval_count": 1,
            "usage_type": "licensed",
        },
        "type": "recurring",
        "unit_amount": entry["unit_amount"],
        "unit_amount_decimal": str(entry["unit_amount"]),
    }
