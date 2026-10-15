"""Real Postgres/Redis prove activation never deletes concurrent paid spend."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from prisma.models import SubscriptionTrial, User

from backend.copilot.rate_limit import (
    RateLimitUnavailable,
    SubscriptionTier,
    check_rate_limit,
    record_cost_usage,
)
from backend.copilot.trial_cost_context import (
    capture_cost_context,
    restore_cost_context,
    trial_cost_context,
)
from backend.copilot.usage_activation import usage_keys, verify_activation_usage
from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.db import execute_raw_with_schema, transaction
from backend.data.pro_activation import get_usage_activation_state
from backend.data.redis_client import get_redis_async
from backend.data.subscription_trial import get_subscription_trial

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


async def publish(user_id: str, generation: str) -> None:
    async with transaction() as tx:
        await execute_raw_with_schema(
            """INSERT INTO {schema_prefix}"PaidUsageActivation"
            ("id", "userId", "stripeSubscriptionId", "stripeInvoiceId", "readyAt")
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)""",
            generation,
            user_id,
            f"sub_{generation}",
            f"in_{generation}",
            client=tx,
        )
        await tx.user.update(where={"id": user_id}, data={"subscriptionTier": "PRO"})
        await tx.subscriptiontrial.update(
            where={"userId": user_id},
            data={"status": "active", "convertedAt": datetime.now(UTC)},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("trial_spent", [40, 100])
async def test_delayed_trial_children_and_retries_preserve_fresh_paid_usage(
    enrollment, trial_spent
):
    now = datetime.now(UTC)
    user_id = enrollment.user_id
    generation = str(uuid4())
    redis = await get_redis_async()
    keys = [*usage_keys(user_id, None, now), *usage_keys(user_id, generation, now)]
    release = asyncio.Event()
    await User.prisma().update(
        where={"id": user_id}, data={"subscriptionTier": "TRIAL"}
    )
    await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={
            "status": "trialing",
            "consumedAt": now,
            "cardVerifiedAt": now,
            "startedAt": now,
            "endsAt": now + timedelta(days=7),
        },
    )

    async def late_child():
        await release.wait()
        await record_cost_usage(user_id, 17, skip_daily=True)

    child = None
    try:
        async with trial_cost_context(user_id):
            await record_cost_usage(user_id, trial_spent)
            child = asyncio.create_task(late_child())
        await publish(user_id, generation)
        state = await get_usage_activation_state(user_id)
        assert state.ready and state.generation == generation and state.trial_id is None
        release.set()
        async with trial_cost_context(user_id):
            await asyncio.gather(
                child,
                *(record_cost_usage(user_id, 3) for _ in range(20)),
                *(verify_activation_usage(user_id, generation) for _ in range(20)),
            )
        values = [int(await redis.get(key) or 0) for key in keys]
        assert values == [trial_spent, trial_spent + 17, 60, 60]
        trial = await get_subscription_trial(user_id)
        assert trial is not None and trial.cost_microdollars == trial_spent + 17
        assert trial.consumed_at is not None and trial.converted_at is not None
        assert all(
            ttl > 0 for ttl in await asyncio.gather(*(redis.ttl(key) for key in keys))
        )
    finally:
        release.set()
        if child is not None:
            await child
        for key in keys:
            await redis.delete(key)


@pytest.mark.asyncio
async def test_unpaid_work_keeps_legacy_namespace_after_initial_signup(enrollment):
    user_id, generation = enrollment.user_id, str(uuid4())
    redis = await get_redis_async()
    keys = [
        *usage_keys(user_id, None, datetime.now(UTC)),
        *usage_keys(user_id, generation, datetime.now(UTC)),
    ]
    try:
        old = await capture_cost_context(user_id)
        await publish(user_id, generation)
        with restore_cost_context(user_id, old):
            await record_cost_usage(user_id, 50)
        await record_cost_usage(user_id, 7)
        assert [int(await redis.get(key) or 0) for key in keys] == [50, 50, 7, 7]
    finally:
        for key in keys:
            await redis.delete(key)


@pytest.mark.asyncio
async def test_activation_between_limits_and_enforcement_fails_closed(enrollment):
    await publish(enrollment.user_id, str(uuid4()))
    with pytest.raises(RateLimitUnavailable, match="changed"):
        await check_rate_limit(
            enrollment.user_id,
            100_000_000,
            100_000_000,
            expected_tier=SubscriptionTier.TRIAL,
        )
