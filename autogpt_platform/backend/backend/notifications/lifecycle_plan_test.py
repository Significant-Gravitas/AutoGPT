import pytest

from backend.notifications import lifecycle_plan


@pytest.mark.parametrize(
    "currency,symbol,amount",
    [
        ("usd", "$", "$50.00"),
        ("USD", "$", "$50.00"),
        ("eur", "€", "€50.00"),
        ("GBP", "£", "£50.00"),
        ("cad", "", "50.00"),
        ("jpy", "", "5,000"),
    ],
)
def test_currency_symbol_matches_formatted_amount(currency, symbol, amount):
    assert lifecycle_plan.currency_symbol(currency) == symbol
    assert lifecycle_plan.format_amount(5000, currency) == amount
