"""A real user on a disposable database, for the usage clock's SQL and Redis."""

import os
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import CreditTransaction, User, UserBalance

from backend.data import db

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable trial database",
)


@pytest_asyncio.fixture
async def billing_user():
    """The id of a fresh user with a Stripe customer, removed afterwards with
    everything that cascades from it."""
    target = urlparse(db.DATABASE_URL)
    local = (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    )
    ci = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and (target.hostname, target.port) == ("localhost", 5432)
        and target.path in ("/postgres", "/ci_copilot")
    )
    assert local or ci, "These tests require an approved disposable database"
    owns_connection = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "stripeCustomerId": f"cus_{user_id}",
            "subscriptionTier": "PRO",
        }
    )
    yield user_id
    await remove_user(user_id)
    if owns_connection:
        await db.disconnect()


async def remove_user(user_id: str) -> None:
    """Delete the user; the credit ledger doesn't cascade, so it goes first."""
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await UserBalance.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})
