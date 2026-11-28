"""Prices as the vendors we sell through take them: whole cents."""

import math
from decimal import ROUND_HALF_UP, Decimal


def price_in_cents(price: float) -> int:
    """``price`` in whole cents, rounded half up; refuses anything that
    rounds to under one cent, which would list the item for nothing.

    Read through its decimal text, so 1.005 is 101 cents as written rather
    than 100 from its binary float.
    """
    if not math.isfinite(price):
        raise ValueError("The price must be a number, e.g. 3.99.")
    cents = int((Decimal(str(price)) * 100).quantize(Decimal(1), ROUND_HALF_UP))
    if cents < 1:
        raise ValueError("The price must be at least 0.01.")
    return cents


def cents_text(cents: int) -> str:
    """Whole cents as a two-decimal amount: 399 is "3.99"."""
    return f"{cents // 100}.{cents % 100:02d}"
