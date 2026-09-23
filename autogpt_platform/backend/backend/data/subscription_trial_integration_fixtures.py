import os
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import SubscriptionTrialClaim, User

from backend.data import db
from backend.data.subscription_trial import reserve_subscription_trial
from backend.data.subscription_trial_config import AcceptedTrialOffer

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable trial database",
)


@pytest_asyncio.fixture
async def enrollment():
    target = urlparse(db.DATABASE_URL)
    local = (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    )
    ci = os.environ.get("GITHUB_ACTIONS") == "true" and (
        target.hostname,
        target.port,
        target.path,
    ) == ("localhost", 5432, "/postgres")
    assert (
        local or ci
    ), "Trial integration tests require an approved disposable database"
    owns_connection = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    trial = await reserve_subscription_trial(
        user_id,
        AcceptedTrialOffer(
            version="trial-test-v1",
            new_users_from=datetime.now(UTC),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=100,
            weekly_cost_limit=100,
            total_cost_limit=100,
            onboarding_credit_amount=300,
            price_id="price_test",
            unit_amount=2000,
            currency="usd",
        ),
        "cus_test",
        "https://example.com/ok",
        "https://example.com/no",
        {},
    )
    yield trial
    await User.prisma().delete_many(where={"id": user_id})
    await SubscriptionTrialClaim.prisma().delete_many(
        where={"trialId": {"startsWith": trial.id}}
    )
    if owns_connection:
        await db.disconnect()
