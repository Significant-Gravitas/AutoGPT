from types import SimpleNamespace

import pytest_mock
import stripe

from .credits import _get_stripe_price_amount


async def test_a_stripe_error_is_not_cached_as_a_zero_price(
    mocker: pytest_mock.MockFixture,
) -> None:
    _get_stripe_price_amount.cache_clear()
    mocker.patch.object(
        stripe.Price,
        "retrieve",
        side_effect=[stripe.StripeError("network"), SimpleNamespace(unit_amount=2000)],
    )

    await _get_stripe_price_amount("price_pro")
    assert await _get_stripe_price_amount("price_pro") == 2000
