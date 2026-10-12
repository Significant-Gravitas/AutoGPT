"""Runs the lifecycle sweep's user query against the real database: keyset
paging over more than one batch, and the billing-history filter."""

import uuid

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial, User

from backend.data.posthog_lifecycle import LifecycleUser
from backend.data.posthog_lifecycle_sync import _lifecycle_user_batch
from backend.util.json import SafeJson


@pytest.fixture
async def lifecycle_users():
    """Five users whose ids sort after every uuid, so paging starts at them.

    1 and 4 have a Stripe customer, 3 has only a trial row, 2 and 5 have
    neither.
    """
    prefix = f"zz-lifecycle-{uuid.uuid4()}-"
    ids = [f"{prefix}{n}" for n in range(1, 6)]
    customers = {ids[0]: f"cus_{prefix}1", ids[3]: f"cus_{prefix}4"}
    for user_id in ids:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"{user_id}@example.com",
                "stripeCustomerId": customers.get(user_id),
                "subscriptionTier": (
                    SubscriptionTier.PRO
                    if user_id == ids[0]
                    else SubscriptionTier.NO_TIER
                ),
            }
        )
    await SubscriptionTrial.prisma().create(
        data={
            "userId": ids[2],
            "offer": SafeJson({}),
            "stripeCustomerId": f"cus_{prefix}3",
            "checkoutSuccessUrl": "https://example.com/ok",
            "checkoutCancelUrl": "https://example.com/no",
        }
    )
    yield prefix, ids
    await User.prisma().delete_many(where={"id": {"in": ids}})


async def _page_through(prefix: str, all_users: bool) -> list[list[LifecycleUser]]:
    batches: list[list[LifecycleUser]] = []
    after_id = prefix
    while batch := await _lifecycle_user_batch(after_id, 2, all_users):
        ours = [user for user in batch if user.id.startswith(prefix)]
        if ours:
            batches.append(ours)
        if len(ours) < len(batch):
            break
        after_id = batch[-1].id
    return batches


@pytest.mark.asyncio(loop_scope="session")
async def test_batches_page_over_users_with_billing_history(lifecycle_users):
    prefix, ids = lifecycle_users

    batches = await _page_through(prefix, all_users=False)

    assert [[user.id for user in batch] for batch in batches] == [
        [ids[0], ids[2]],
        [ids[3]],
    ]
    first = batches[0][0]
    assert first.stripe_customer_id == f"cus_{prefix}1"
    assert first.subscription_tier == SubscriptionTier.PRO
    assert first.created_at is not None
    assert batches[0][1].stripe_customer_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_all_users_includes_users_without_billing_history(lifecycle_users):
    prefix, ids = lifecycle_users

    batches = await _page_through(prefix, all_users=True)

    assert [[user.id for user in batch] for batch in batches] == [
        ids[0:2],
        ids[2:4],
        ids[4:5],
    ]
    assert batches[0][1].subscription_tier == SubscriptionTier.NO_TIER
