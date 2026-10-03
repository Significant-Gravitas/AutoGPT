"""DB-backed regression test for SECRT-2770.

An admin grants PRO to a user with no Stripe customer, the user opens
Settings > Billing (which requests a billing-portal link), and the Stripe
reconciliation sweep runs. The grant must survive: opening the billing page
must not create a Stripe customer, so the sweep never sees the user as
Stripe-billed.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.copilot.rate_limit import get_user_tier, set_user_tier
from backend.data.credit import UserCredit, _is_stripe_reconcilable
from backend.data.stripe_reconciliation import reconcile_all_stripe_tiers
from backend.data.user import get_user_by_id
from backend.util.json import SafeJson


@pytest.fixture
async def comped_user_id():
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"comped-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})


def _empty_stripe_pages(mocker: pytest_mock.MockFixture) -> None:
    """Stripe reports no active or trialing subscriptions at all."""

    def _list(*, status: str, limit: int, starting_after: str | None = None):
        page = MagicMock()
        page.data = []
        page.has_more = False
        return page

    mocker.patch(
        "backend.data.stripe_reconciliation.stripe.Subscription.list_async",
        side_effect=_list,
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={},
    )


def _scope_sweep_to(mocker: pytest_mock.MockFixture, user_id: str) -> AsyncMock:
    """Run the sweep's real candidate query, narrowed to this test's user so
    other rows in the shared test database are left untouched.

    ``User.prisma`` is one shared classmethod, so every other call keeps
    going to the real client through the proxy; only ``find_many`` is
    narrowed, and only while the sweep runs.
    """
    real_prisma = User.prisma
    spy = AsyncMock()

    def scoped_prisma(*args, **kwargs):
        client = real_prisma(*args, **kwargs)

        async def find_many(*, where, **extra):
            return await client.find_many(where={**where, "id": user_id}, **extra)

        spy.side_effect = find_many
        proxy = MagicMock(wraps=client)
        proxy.find_many = spy
        return proxy

    mocker.patch.object(User, "prisma", side_effect=scoped_prisma)
    return spy


@pytest.mark.asyncio(loop_scope="session")
async def test_admin_grant_survives_opening_billing_and_the_sweep(
    server, mocker: pytest_mock.MockFixture, comped_user_id: str
) -> None:
    mocker.patch("backend.copilot.rate_limit.schedule_posthog_lifecycle_sync")
    mocker.patch(
        "backend.copilot.rate_limit._drift_check_background", new_callable=AsyncMock
    )
    customer_create = mocker.patch(
        "backend.data.credit.stripe.Customer.create_async", new_callable=AsyncMock
    )
    portal_create = mocker.patch(
        "backend.data.credit.stripe.billing_portal.Session.create_async",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.alert_tier_reconciliation_discrepancy",
        new_callable=AsyncMock,
    )
    _empty_stripe_pages(mocker)

    # 1. Admin grants PRO through Admin > Rate limits.
    await set_user_tier(comped_user_id, SubscriptionTier.PRO)
    assert await get_user_tier(comped_user_id) == SubscriptionTier.PRO

    # 2. The user opens Settings > Billing, which requests a portal link.
    assert await UserCredit.create_billing_portal_session(comped_user_id) is None
    customer_create.assert_not_awaited()
    portal_create.assert_not_awaited()
    user = await get_user_by_id(comped_user_id)
    assert user.stripe_customer_id is None
    assert not _is_stripe_reconcilable(user)

    # 3. The sweep runs with no active Stripe subscriptions anywhere.
    find_many = _scope_sweep_to(mocker, comped_user_id)
    summary = await reconcile_all_stripe_tiers()
    mocker.stopall()

    # 4. The grant is untouched: the user never became a sweep candidate.
    find_many.assert_awaited_once()
    assert find_many.await_args.kwargs["where"]["stripeCustomerId"] == {"not": None}
    assert summary.candidate_users == 0
    assert summary.downgrades == 0
    assert summary.discrepancies == []
    assert await get_user_tier(comped_user_id) == SubscriptionTier.PRO
    row = await User.prisma().find_unique(where={"id": comped_user_id})
    assert row is not None
    assert row.subscriptionTier == SubscriptionTier.PRO
    assert row.stripeCustomerId is None
