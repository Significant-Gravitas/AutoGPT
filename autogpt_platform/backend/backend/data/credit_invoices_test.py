"""Unit tests for UserCredit.list_invoices.

These tests intentionally mock the user lookup and the Stripe SDK so they
don't need a database or network — the goal is to lock in the contract that
``list_invoices``:

  1. Returns ``[]`` and never calls Stripe when the user has no
     ``stripe_customer_id`` (avoids creating orphaned customers for every
     beta user that opens the billing page).
  2. Maps a Stripe invoice list into ``InvoiceListItem`` rows, exposing
     ``total_cents`` (from ``invoice.total``) so open/unpaid invoices show
     the correct amount.
  3. Degrades to ``[]`` on any ``stripe.StripeError`` so a Stripe outage
     surfaces as an empty card, not a 500.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe

from backend.data import credit as credit_module
from backend.data.credit import InvoiceListItem, UserCredit


def _make_user(stripe_customer_id: str | None):
    user = MagicMock()
    user.stripe_customer_id = stripe_customer_id
    user.name = "Test User"
    user.email = "test@example.com"
    return user


def _make_stripe_invoice(**overrides):
    """Build a MagicMock that quacks like ``stripe.Invoice``."""
    invoice = MagicMock()
    defaults = {
        "id": "in_test_123",
        "number": "INV-001",
        "created": int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()),
        "total": 2000,
        "amount_paid": 0,
        "currency": "USD",
        "status": "open",
        "description": "Subscription",
        "hosted_invoice_url": "https://invoice.stripe.com/i/test",
        "invoice_pdf": "https://invoice.stripe.com/i/test/pdf",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(invoice, key, value)
    return invoice


class TestListInvoices:
    @pytest.mark.asyncio
    async def test_returns_empty_when_user_has_no_stripe_customer(self):
        """No customer_id → no Stripe call (would otherwise create one)."""
        user_credit = UserCredit()
        with (
            patch.object(
                credit_module,
                "get_user_by_id",
                AsyncMock(return_value=_make_user(None)),
            ),
            patch.object(credit_module.stripe.Invoice, "list") as mock_list,
        ):
            result = await user_credit.list_invoices("user-1")

        assert result == []
        mock_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_on_stripe_error(self):
        user_credit = UserCredit()
        with (
            patch.object(
                credit_module,
                "get_user_by_id",
                AsyncMock(return_value=_make_user("cus_test")),
            ),
            patch.object(
                credit_module.stripe.Invoice,
                "list",
                side_effect=stripe.StripeError("boom"),
            ),
        ):
            result = await user_credit.list_invoices("user-1")

        assert result == []

    @pytest.mark.asyncio
    async def test_maps_stripe_invoices_using_total_for_amount(self):
        """``total_cents`` is sourced from ``invoice.total`` so open/unpaid
        invoices don't render as $0.00."""
        user_credit = UserCredit()
        open_invoice = _make_stripe_invoice(
            id="in_open",
            number="INV-002",
            total=2500,
            amount_paid=0,
            status="open",
        )
        paid_invoice = _make_stripe_invoice(
            id="in_paid",
            number="INV-003",
            total=4000,
            amount_paid=4000,
            status="paid",
            currency="usd",
        )
        stripe_response = MagicMock(data=[open_invoice, paid_invoice])

        with (
            patch.object(
                credit_module,
                "get_user_by_id",
                AsyncMock(return_value=_make_user("cus_test")),
            ),
            patch.object(
                credit_module.stripe.Invoice,
                "list",
                return_value=stripe_response,
            ) as mock_list,
        ):
            result = await user_credit.list_invoices("user-1", limit=24)

        assert mock_list.call_args.kwargs == {"customer": "cus_test", "limit": 24}
        assert len(result) == 2
        assert all(isinstance(row, InvoiceListItem) for row in result)

        first, second = result
        assert (first.id, first.total_cents, first.amount_paid_cents) == (
            "in_open",
            2500,
            0,
        )
        assert first.currency == "usd", "currency should be lowercased"
        assert first.status == "open"
        assert first.hosted_invoice_url == "https://invoice.stripe.com/i/test"
        assert first.invoice_pdf_url == "https://invoice.stripe.com/i/test/pdf"

        assert (second.id, second.total_cents, second.amount_paid_cents) == (
            "in_paid",
            4000,
            4000,
        )
        assert second.status == "paid"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "requested_limit,expected_limit",
        [(0, 1), (5, 5), (250, 100)],
    )
    async def test_limit_is_clamped_to_stripe_bounds(
        self, requested_limit, expected_limit
    ):
        user_credit = UserCredit()
        stripe_response = MagicMock(data=[])
        with (
            patch.object(
                credit_module,
                "get_user_by_id",
                AsyncMock(return_value=_make_user("cus_test")),
            ),
            patch.object(
                credit_module.stripe.Invoice,
                "list",
                return_value=stripe_response,
            ) as mock_list,
        ):
            await user_credit.list_invoices("user-1", limit=requested_limit)

        assert mock_list.call_args.kwargs["limit"] == expected_limit
