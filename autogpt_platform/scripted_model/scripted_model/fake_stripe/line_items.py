"""Inline prices (``price_data``) on a checkout's line items, checked as
Stripe checks them: a product, and an amount."""

from decimal import Decimal, InvalidOperation
from typing import Any

from .errors import ApiError, missing

Params = dict[str, Any]


def product_name(price_data: Params, where: str) -> str:
    """An inline price needs a product: an existing one, or ``product_data``
    with a name."""
    product_data = price_data.get("product_data")
    if isinstance(product_data, dict):
        if not product_data.get("name"):
            raise missing(f"{where}[product_data][name]")
        return str(product_data["name"])
    if price_data.get("product"):
        return str(price_data["product"])
    raise ApiError(
        400,
        "You must specify either `product` or `product_data` when creating a" " price.",
        param=f"{where}[product_data]",
    )


def unit_price(price_data: Params, where: str) -> Decimal:
    """The price of one unit in the currency's smallest unit: ``unit_amount``,
    else ``unit_amount_decimal``, kept exact so the line total is rounded
    once, after the quantity. One of them is required, and 0 is a price."""
    amount = price_data.get("unit_amount")
    if amount is not None:
        if not str(amount).isdigit():
            raise ApiError(
                400, "Invalid integer: " + str(amount), param=f"{where}[unit_amount]"
            )
        return Decimal(int(amount))
    decimal = price_data.get("unit_amount_decimal")
    if decimal is None:
        raise missing(f"{where}[unit_amount]")
    try:
        value = Decimal(str(decimal))
    except InvalidOperation:
        value = Decimal(-1)
    if not value.is_finite() or value < 0:
        raise ApiError(
            400,
            "Invalid decimal: " + str(decimal),
            param=f"{where}[unit_amount_decimal]",
        )
    return value
