import math

import pytest

from backend.util.money import cents_text, price_in_cents


@pytest.mark.parametrize(
    "price, cents",
    [
        (3.99, 399),
        (0.01, 1),
        (0.005, 1),  # half a cent rounds up to the smallest price
        (1.005, 101),  # as written, not the float's 1.00499...
        (2.0, 200),
        (19.999, 2000),
    ],
)
def test_a_price_is_whole_cents_rounded_half_up(price: float, cents: int):
    assert price_in_cents(price) == cents


@pytest.mark.parametrize("price", [0, 0.004, 0.0049, -0.01, -3.99])
def test_a_price_under_one_cent_is_refused(price: float):
    with pytest.raises(ValueError, match="at least 0.01"):
        price_in_cents(price)


@pytest.mark.parametrize("price", [math.nan, math.inf, -math.inf])
def test_a_price_that_is_not_a_number_is_refused(price: float):
    with pytest.raises(ValueError, match="must be a number"):
        price_in_cents(price)


@pytest.mark.parametrize("cents, text", [(399, "3.99"), (1, "0.01"), (2000, "20.00")])
def test_cents_read_as_a_two_decimal_amount(cents: int, text: str):
    assert cents_text(cents) == text
