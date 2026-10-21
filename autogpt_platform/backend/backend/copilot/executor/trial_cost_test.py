"""Exercise delayed usage through the real turn execution boundary."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatSession
from backend.copilot.rate_limit import SubscriptionTier
from backend.copilot.response_model import StreamStatus
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.trial_cost_context import get_trial_cost_context


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sdk", [True, False])
async def test_delayed_trial_cost_and_later_paid_turn_are_separate(use_sdk):
    trial = SimpleNamespace(id="trial-1", active=True, consumed_at=True)
    store = MagicMock()
    store.get_subscription_trial = AsyncMock(return_value=trial)
    store.record_subscription_trial_cost = AsyncMock()
    tier = AsyncMock(return_value=SubscriptionTier.TRIAL)
    release = asyncio.Event()
    pending: list[asyncio.Task] = []

    async def settle():
        await release.wait()
        await persist_and_record_usage(
            session=None,
            user_id="user-1",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.01,
        )

    async def stream(**_kwargs):
        pending.append(asyncio.create_task(settle()))
        yield StreamStatus(message="Done")

    entry = CoPilotExecutionEntry(
        session_id="session-1", turn_id="turn-1", user_id="user-1", message="Hi"
    )
    with (
        patch("backend.data.db_accessors.credit_db", return_value=store),
        patch("backend.copilot.rate_limit.credit_db", return_value=store),
        patch("backend.copilot.rate_limit._fetch_user_tier", tier),
        patch("backend.copilot.rate_limit.get_redis_async", AsyncMock()),
        patch("backend.copilot.rate_limit._incr_counter_atomic", AsyncMock()),
        patch("backend.copilot.token_tracking._schedule_cost_log") as cost_log,
        patch(
            "backend.copilot.model.get_chat_session",
            AsyncMock(return_value=ChatSession.new("user-1", dry_run=False)),
        ),
        patch(
            "backend.copilot.executor.processor.ChatConfig",
            return_value=MagicMock(test_mode=False),
        ),
        patch(
            "backend.copilot.executor.processor.resolve_use_sdk",
            AsyncMock(return_value=use_sdk),
        ),
        patch(
            "backend.copilot.executor.processor._building_mode_forces_sdk",
            AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.executor.processor.sdk_service.stream_chat_completion_sdk",
            stream,
        ),
        patch(
            "backend.copilot.executor.processor.stream_chat_completion_baseline", stream
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.stream_and_publish",
            lambda **kw: kw["stream"],
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.publish_chunk",
            AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            AsyncMock(),
        ),
    ):
        processor = CoPilotProcessor()
        log = CoPilotLogMetadata(logger=logging.getLogger("trial-cost-test"))
        try:
            await processor._execute_async(entry, threading.Event(), MagicMock(), log)
            trial.active = False
            tier.return_value = SubscriptionTier.PRO
            release.set()
            await asyncio.gather(*pending)
            await processor._execute_async(
                entry.model_copy(update={"turn_id": "turn-2"}),
                threading.Event(),
                MagicMock(),
                log,
            )
            await asyncio.gather(*pending)
        finally:
            release.set()
            await asyncio.gather(*pending)

    store.record_subscription_trial_cost.assert_awaited_once_with(
        "user-1", 10_000, trial_id="trial-1"
    )
    assert [
        call.args[0].metadata["subscription_trial_id"]
        for call in cost_log.call_args_list
    ] == ["trial-1", None]
    tier.assert_not_awaited()
    assert get_trial_cost_context("user-1") is None
