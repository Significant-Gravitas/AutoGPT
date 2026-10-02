"""Unit tests for ``UserCreditBase.create_billing_portal_session``.

Settings > Billing requests the portal link on every page load, so the lookup
must never provision a Stripe Customer as a side effect (SECRT-2770). Doing so
made admin-granted plans look Stripe-billed to the reconciliation sweep, which
then revoked them to NO_TIER.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.credit import UserCredit


def _make_user(stripe_customer_id: str | None):
    user = MagicMock()
    user.stripe_customer_id = stripe_customer_id
    user.name = "Test User"
    user.email = "test@example.com"
    return user


class TestCreateBillingPortalSession:
    @pytest.mark.asyncio
    async def test_no_customer_returns_none_without_touching_stripe(self):
        """No customer → ``None``; neither Customer.create nor a portal session
        is requested, and the user row is left alone."""
        update = AsyncMock()
        with (
            patch(
                "backend.data.credit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=_make_user(None),
            ),
            patch(
                "backend.data.credit.stripe.Customer.create_async",
                new_callable=AsyncMock,
            ) as customer_create,
            patch(
                "backend.data.credit.stripe.billing_portal.Session.create_async",
                new_callable=AsyncMock,
            ) as portal_create,
            patch(
                "backend.data.credit.User.prisma",
                return_value=MagicMock(update=update),
            ),
        ):
            url = await UserCredit.create_billing_portal_session("user-1")

        assert url is None
        customer_create.assert_not_awaited()
        portal_create.assert_not_awaited()
        update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existing_customer_gets_a_portal_url(self):
        session = MagicMock()
        session.url = "https://billing.stripe.com/session/abc"
        with (
            patch(
                "backend.data.credit.get_user_by_id",
                new_callable=AsyncMock,
                return_value=_make_user("cus_123"),
            ),
            patch(
                "backend.data.credit.stripe.Customer.create_async",
                new_callable=AsyncMock,
            ) as customer_create,
            patch(
                "backend.data.credit.stripe.billing_portal.Session.create_async",
                new_callable=AsyncMock,
                return_value=session,
            ) as portal_create,
        ):
            url = await UserCredit.create_billing_portal_session("user-1")

        assert url == "https://billing.stripe.com/session/abc"
        customer_create.assert_not_awaited()
        portal_create.assert_awaited_once()
        assert portal_create.await_args.kwargs["customer"] == "cus_123"
