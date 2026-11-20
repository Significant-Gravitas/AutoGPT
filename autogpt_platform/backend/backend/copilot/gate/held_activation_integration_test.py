"""Delayed approvals retain trial origin through Redis and the database FIFO."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prisma.enums import ReviewStatus, SubscriptionTier

from backend.copilot import db as chat_db
from backend.copilot import rate_limit, turn_queue
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.executor import utils
from backend.copilot.gate import held
from backend.copilot.gate.held_db_test import (
    _answer,
    _hold,
    _new_session,
    _Post,
    _resolved,
)
from backend.copilot.tools.models import ResponseType, ToolResponseBase
from backend.copilot.trial_cost_context import (
    TrialCostContext,
    capture_cost_context,
    get_trial_cost_context,
)
from backend.copilot.turn_queue_activation_integration_test import (
    _publish_activation,
    _start_trial,
)
from backend.copilot.usage_activation import usage_keys, verify_activation_usage
from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.redis_client import get_redis_async
from backend.data.subscription_trial import get_subscription_trial

enrollment = fixtures.enrollment
usage_now = fixtures.usage_now
pytestmark = fixtures.pytestmark


@pytest.mark.parametrize("paid_tier", [SubscriptionTier.PRO, SubscriptionTier.MAX])
@pytest.mark.parametrize("include_paid_call", [False, True])
async def test_queued_approvals_charge_their_original_allowances(
    enrollment, usage_now, paid_tier, include_paid_call, monkeypatch
):
    user_id, generation = enrollment.user_id, str(uuid4())
    await _start_trial(enrollment.id, user_id)
    monkeypatch.setattr(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(turn_queue, "track_user_message", MagicMock())
    monkeypatch.setattr(
        rate_limit, "get_feature_flag_value", AsyncMock(return_value=None)
    )
    session = await _new_session(user_id)
    tool = _Post()
    origins = []

    async def execute(user_id, session, **kwargs):
        origins.append(await capture_cost_context(user_id))
        await asyncio.create_task(rate_limit.record_cost_usage(user_id, 17))
        return ToolResponseBase(type=ResponseType.ERROR, message="delivered")

    monkeypatch.setattr(tool, "_execute", execute)
    monkeypatch.setattr("backend.copilot.tools.get_tool", lambda _: tool)
    redis = await get_redis_async()
    keys = [
        *usage_keys(user_id, None, usage_now),
        *usage_keys(user_id, generation, usage_now),
        held._key(session.session_id),
    ]
    try:
        await rate_limit.record_cost_usage(user_id, 40)
        trial_review = await _hold(session, user_id, "trial action")
        await _publish_activation(user_id, generation, paid_tier)
        await rate_limit.record_cost_usage(user_id, 7)
        review_ids = [trial_review]
        if include_paid_call:
            review_ids.append(await _hold(session, user_id, "paid action"))
        for review_id in review_ids:
            await _answer(review_id, ReviewStatus.APPROVED)

        @asynccontextmanager
        async def full(*args, **kwargs):
            raise ConcurrentTurnLimitError()
            yield

        monkeypatch.setattr("backend.copilot.active_turns.acquire_turn_slot", full)
        await held.wake(user_id, session.session_id)
        queued = await chat_db.get_latest_user_message_in_session(session.session_id)
        expected_paid = TrialCostContext(
            user_id=user_id, trial_id=None, generation=generation
        )
        expected_trial = TrialCostContext(user_id=user_id, trial_id=enrollment.id)
        wake_origin = expected_paid if include_paid_call else expected_trial
        assert queued is not None and queued.metadata is not None
        assert queued.metadata["cost_context"] == wake_origin.model_dump(mode="json")

        async def dispatch(*args, **kwargs):
            assert await capture_cost_context(user_id) == wake_origin
            results = await asyncio.gather(
                _resolved(user_id, session), _resolved(user_id, session)
            )
            assert sum(map(len, results)) == len(review_ids)

        monkeypatch.setattr(utils, "dispatch_turn", dispatch)
        assert await turn_queue.dispatch_next_for_user(user_id)
        assert len(origins) == len(review_ids)
        assert expected_trial in origins
        assert (expected_paid in origins) == include_paid_call
        assert get_trial_cost_context(user_id) is None
        assert await _resolved(user_id, session) == []
        await verify_activation_usage(user_id, generation)
        paid_usage = 24 if include_paid_call else 7
        assert [int(await redis.get(key) or 0) for key in keys[:4]] == [
            57,
            57,
            paid_usage,
            paid_usage,
        ]
        trial = await get_subscription_trial(user_id)
        assert trial is not None and trial.cost_microdollars == 57
        assert trial.consumed_at is not None and trial.converted_at is not None
    finally:
        for key in keys:
            await redis.delete(key)
