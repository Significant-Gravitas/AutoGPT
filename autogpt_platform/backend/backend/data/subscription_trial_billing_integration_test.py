"""Selection boundaries for billing events, against an explicitly disposable DB."""

import os
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import SubscriptionTrial, User

from backend.data import db
from backend.data.subscription_trial_billing import get_trial_billing_targets
from backend.util.json import SafeJson

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable trial database",
)


@pytest_asyncio.fixture
async def customer_id():
    target = urlparse(db.DATABASE_URL)
    assert (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    ) or (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and (target.hostname, target.port, target.path)
        == ("localhost", 5432, "/postgres")
    )
    owns_connection = not db.is_connected()
    await db.connect()
    customer = f"cus_billing_test_{uuid4()}"
    yield customer
    rows = await SubscriptionTrial.prisma().find_many(
        where={"stripeCustomerId": customer}
    )
    await User.prisma().delete_many(where={"id": {"in": [row.userId for row in rows]}})
    if owns_connection:
        await db.disconnect()


async def create_trial(
    customer_id,
    *,
    consumed=True,
    converted=False,
    subscription=True,
    current_customer=None,
    enterprise=False,
):
    user_id = str(uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "stripeCustomerId": current_customer or customer_id,
            "subscriptionTier": "ENTERPRISE" if enterprise else "TRIAL",
        }
    )
    return await SubscriptionTrial.prisma().create(
        data={
            "userId": user_id,
            "stripeCustomerId": customer_id,
            "stripeSubscriptionId": f"sub_{uuid4()}" if subscription else None,
            "offer": SafeJson({"validation": "billing-target-selection"}),
            "checkoutSuccessUrl": "https://example.com/success",
            "checkoutCancelUrl": "https://example.com/cancel",
            "consumedAt": datetime.now(UTC) if consumed else None,
            "convertedAt": datetime.now(UTC) if converted else None,
        }
    )


@pytest.mark.asyncio
async def test_billing_targets_exclude_unowned_pending_converted_and_enterprise(
    customer_id,
):
    expected = await create_trial(customer_id)
    await create_trial(customer_id, consumed=False)
    await create_trial(customer_id, converted=True)
    await create_trial(customer_id, subscription=False)
    await create_trial(customer_id, current_customer="cus_changed_mapping")
    await create_trial(customer_id, enterprise=True)
    rows = await get_trial_billing_targets([customer_id])
    assert [row.id for row in rows] == [expected.id]
    assert rows[0].user_id == expected.userId
    assert rows[0].subscription_id == expected.stripeSubscriptionId
    assert await get_trial_billing_targets(["cus_not_this_customer"]) == []
    assert await get_trial_billing_targets([]) == []


@pytest.mark.asyncio
async def test_billing_targets_keyset_pagination_does_not_skip_trials(customer_id):
    rows = [await create_trial(customer_id) for _ in range(101)]
    first = await get_trial_billing_targets([customer_id])
    assert len(first) == 100
    second = await get_trial_billing_targets([customer_id], first[-1].id)
    assert len(second) == 1
    assert [row.id for row in first + second] == sorted(row.id for row in rows)
    assert await get_trial_billing_targets([customer_id], second[-1].id) == []
