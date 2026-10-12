"""Persisted FIFO work retains trial attribution across paid activation."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial, User

from backend.copilot import db as chat_db
from backend.copilot import rate_limit, turn_queue
from backend.copilot.executor import utils as executor_utils
from backend.copilot.trial_cost_context import (
    TrialCostContext,
    capture_cost_context,
    get_trial_cost_context,
)
from backend.copilot.usage_activation import usage_keys, verify_activation_usage
from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.db import execute_raw_with_schema, transaction
from backend.data.redis_client import get_redis_async
from backend.data.subscription_trial import get_subscription_trial

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


@pytest.mark.asyncio
@pytest.mark.parametrize("paid_tier", [SubscriptionTier.PRO, SubscriptionTier.MAX])
async def test_persisted_trial_turn_keeps_trial_costs_after_activation_and_retry(
    enrollment, paid_tier, monkeypatch
):
    user_id, session_id, generation = enrollment.user_id, str(uuid4()), str(uuid4())
    await _start_trial(enrollment.id, user_id)
    await chat_db.create_chat_session(session_id, user_id)
    monkeypatch.setattr(turn_queue, "track_user_message", MagicMock())
    monkeypatch.setattr(
        rate_limit, "get_feature_flag_value", AsyncMock(return_value=None)
    )
    dispatched: list[TrialCostContext] = []

    async def dispatch(*args, **kwargs):
        dispatched.append(await capture_cost_context(user_id))
        if len(dispatched) == 1:
            raise RuntimeError("Queue transport unavailable")
        await asyncio.create_task(rate_limit.record_cost_usage(user_id, 17))

    monkeypatch.setattr(executor_utils, "dispatch_turn", dispatch)
    redis = await get_redis_async()
    keys = [
        *usage_keys(user_id, None, datetime.now(UTC)),
        *usage_keys(user_id, generation, datetime.now(UTC)),
    ]
    try:
        await rate_limit.record_cost_usage(user_id, 40)
        await turn_queue.enqueue_turn(
            user_id=user_id, session_id=session_id, message="Queued during trial"
        )
        pending = await chat_db.get_latest_user_message_in_session(session_id)
        assert pending is not None and pending.metadata is not None
        expected = TrialCostContext(user_id=user_id, trial_id=enrollment.id)
        assert pending.metadata["cost_context"] == expected.model_dump(mode="json")
        assert await chat_db.get_chat_session_status(session_id) == "queued"

        await _publish_activation(user_id, generation, paid_tier)
        await rate_limit.record_cost_usage(user_id, 7)
        with pytest.raises(RuntimeError, match="Queue transport unavailable"):
            await turn_queue.dispatch_next_for_user(user_id)
        assert await chat_db.get_chat_session_status(session_id) == "queued"
        assert get_trial_cost_context(user_id) is None

        await rate_limit.record_cost_usage(user_id, 11)
        await verify_activation_usage(user_id, generation)
        assert await turn_queue.dispatch_next_for_user(user_id)
        assert await chat_db.get_chat_session_status(session_id) == "running"
        assert dispatched == [expected, expected]
        assert get_trial_cost_context(user_id) is None
        await verify_activation_usage(user_id, generation)
        assert [int(await redis.get(key) or 0) for key in keys] == [57, 57, 18, 18]
        trial = await get_subscription_trial(user_id)
        assert trial is not None and trial.cost_microdollars == 57
        assert trial.consumed_at is not None and trial.converted_at is not None
    finally:
        for key in keys:
            await redis.delete(key)


async def _start_trial(trial_id: str, user_id: str) -> None:
    now = datetime.now(UTC)
    await User.prisma().update(
        where={"id": user_id}, data={"subscriptionTier": SubscriptionTier.TRIAL}
    )
    await SubscriptionTrial.prisma().update(
        where={"id": trial_id},
        data={
            "status": "trialing",
            "consumedAt": now,
            "cardVerifiedAt": now,
            "startedAt": now,
            "endsAt": now + timedelta(days=7),
        },
    )


async def _publish_activation(
    user_id: str, generation: str, tier: SubscriptionTier
) -> None:
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
        await tx.user.update(where={"id": user_id}, data={"subscriptionTier": tier})
        await tx.subscriptiontrial.update(
            where={"userId": user_id},
            data={"status": "active", "convertedAt": datetime.now(UTC)},
        )
