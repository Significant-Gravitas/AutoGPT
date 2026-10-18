"""The PostHog audience backfill's reads against the real tables: every account
with what its plan needs, and every account's sign-in providers. The unit tests
patch both out, so a renamed column would leave them green."""

from uuid import uuid4

import pytest
from prisma.models import AuthAccount, AuthUser, User

from backend.cli import mailerlite_backfill
from backend.cli import posthog_audience_backfill as cli
from backend.util.test import SpinTestServer


@pytest.fixture
async def account(server: SpinTestServer):
    """An account with an email sign-in, a browser timezone and a Stripe
    customer."""
    user_id = str(uuid4())
    email = f"{user_id}@example.com"
    await AuthUser.prisma().create(
        data={"id": user_id, "name": "PostHog backfill", "email": email}
    )
    await AuthAccount.prisma().create(
        data={
            "id": str(uuid4()),
            "accountId": user_id,
            "providerId": "credential",
            "userId": user_id,
        }
    )
    await User.prisma().create(
        data={
            "id": user_id,
            "email": email,
            "timezone": "Europe/Vienna",
            "stripeCustomerId": f"cus_{user_id}",
        }
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})
    # Takes its AuthAccount with it.
    await AuthUser.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_every_account_is_read_with_what_its_plan_needs(account: str):
    (row,) = [a for a in await cli._accounts() if a.user_id == account]

    assert row.email == f"{account}@example.com"
    assert row.timezone == "Europe/Vienna"
    assert row.stripe_customer_id == f"cus_{account}"
    assert row.created_at is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_each_accounts_sign_in_providers_are_read(account: str):
    providers = await mailerlite_backfill.signin_providers_by_account()

    assert providers[account] == ["credential"]
