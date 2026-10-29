"""Attribution survives the database FIFO before RabbitMQ publication."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.copilot import stream_registry, turn_queue
from backend.copilot.executor import utils
from backend.copilot.model import ChatMessage, ChatSessionInfo
from backend.copilot.tree import root_envelope
from backend.copilot.trial_cost_context import (
    TrialCostContext,
    get_trial_cost_context,
    restore_cost_context,
)
from backend.copilot.usage_activation import UsageActivationUnavailable
from backend.data.pro_activation import UsageActivationState


@pytest.fixture
def fifo(monkeypatch: pytest.MonkeyPatch):
    @asynccontextmanager
    async def lock(_session_id):
        yield True

    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=1)
    db.add_chat_message = AsyncMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.get_latest_user_message_in_session = AsyncMock()
    session = ChatSessionInfo(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.metadata.llm_auth_provider = "platform"
    session.metadata.llm_credential_id = None
    monkeypatch.setattr(turn_queue, "chat_db", lambda: db)
    monkeypatch.setattr(turn_queue, "_get_session_lock", lock)
    monkeypatch.setattr(turn_queue, "invalidate_session_cache", AsyncMock())
    monkeypatch.setattr(turn_queue, "track_user_message", MagicMock())
    monkeypatch.setattr(
        turn_queue, "list_queued_sessions", AsyncMock(return_value=[session])
    )
    monkeypatch.setattr(turn_queue, "is_user_paywalled", AsyncMock(return_value=False))
    monkeypatch.setattr(
        turn_queue, "get_global_rate_limits", AsyncMock(return_value=(100, 1000, None))
    )
    monkeypatch.setattr(turn_queue, "check_rate_limit", AsyncMock())
    monkeypatch.setattr(
        utils,
        "_admitted_turn_envelope",
        AsyncMock(return_value=root_envelope("turn-1", session_id="session-1")),
    )
    monkeypatch.setattr(utils, "release_turn", AsyncMock())
    monkeypatch.setattr(stream_registry, "create_session", AsyncMock())
    monkeypatch.setattr(stream_registry, "delete_session_meta", AsyncMock())
    publisher = AsyncMock()
    monkeypatch.setattr(
        "backend.util.clients.get_async_copilot_queue",
        AsyncMock(return_value=publisher),
    )
    current = AsyncMock(
        return_value=UsageActivationState(
            user_id="user-1", generation="paid-1", tier=SubscriptionTier.PRO, ready=True
        )
    )
    monkeypatch.setattr(
        "backend.copilot.trial_cost_context.get_ready_usage_state", current
    )
    return db, publisher, current


def pending(metadata):
    return ChatMessage(
        id="message-1",
        session_id="session-1",
        role="user",
        content="queued work",
        sequence=1,
        created_at=datetime.now(timezone.utc),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_admission_captures_authoritative_attribution_without_parent(fifo):
    db, _publisher, current = fifo
    await turn_queue.enqueue_turn(
        user_id="user-1", session_id="session-1", message="queued work"
    )
    current.assert_awaited_once_with("user-1")
    assert db.add_chat_message.call_args.kwargs["metadata"]["cost_context"] == {
        "user_id": "user-1",
        "trial_id": None,
        "generation": "paid-1",
    }


@pytest.mark.asyncio
async def test_unavailable_attribution_prevents_queue_admission(fifo):
    db, publisher, current = fifo
    current.side_effect = UsageActivationUnavailable("Usage activation is processing")
    with pytest.raises(UsageActivationUnavailable):
        await turn_queue.enqueue_turn(
            user_id="user-1", session_id="session-1", message="queued work"
        )
    db.add_chat_message.assert_not_awaited()
    db.update_chat_session_status.assert_not_awaited()
    publisher.publish_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("trial_id,generation", [("trial-1", None), (None, "paid-1")])
async def test_queued_turn_keeps_admission_context_across_activation(
    fifo, trial_id, generation
):
    db, publisher, current = fifo
    original = TrialCostContext(
        user_id="user-1", trial_id=trial_id, generation=generation
    )
    with restore_cost_context("user-1", original):
        await turn_queue.enqueue_turn(
            user_id="user-1",
            session_id="session-1",
            message="queued work",
            message_metadata={"cost_context": {"user_id": "forged"}},
        )
    metadata = db.add_chat_message.call_args.kwargs["metadata"]
    assert metadata["cost_context"] == original.model_dump(mode="json")
    db.get_latest_user_message_in_session.return_value = pending(metadata)
    completing_turn = TrialCostContext(
        user_id="user-1", trial_id=None, generation="other-generation"
    )
    with restore_cost_context("user-1", completing_turn):
        assert await turn_queue.dispatch_next_for_user("user-1")
        assert get_trial_cost_context("user-1") == completing_turn
    entry = utils.CoPilotExecutionEntry.model_validate_json(
        publisher.publish_message.call_args.kwargs["message"]
    )
    assert entry.cost_context == original
    assert get_trial_cost_context("user-1") is None
    current.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_uses_persisted_trial_snapshot_after_publish_failure(fifo):
    db, publisher, current = fifo
    snapshot = TrialCostContext(user_id="user-1", trial_id="trial-1")
    db.get_latest_user_message_in_session.return_value = pending(
        {"cost_context": snapshot.model_dump(mode="json")}
    )
    publisher.publish_message.side_effect = [RuntimeError("RabbitMQ blip"), None]
    # A publish that raised may still have reached the broker, so it surfaces
    # as TurnPublishUnconfirmed with the broker error as its cause.
    with pytest.raises(utils.TurnPublishUnconfirmed) as raised:
        await turn_queue.dispatch_next_for_user("user-1")
    assert str(raised.value.__cause__) == "RabbitMQ blip"
    db.update_chat_session_status.assert_any_await(
        session_id="session-1", expect_status="running", status="queued"
    )
    assert get_trial_cost_context("user-1") is None
    assert await turn_queue.dispatch_next_for_user("user-1")
    for call in publisher.publish_message.call_args_list:
        entry = utils.CoPilotExecutionEntry.model_validate_json(call.kwargs["message"])
        assert entry.cost_context == snapshot
    current.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"cost_context": None},
        {"cost_context": {"user_id": "user-1"}},
        {"cost_context": {"user_id": "other-user", "trial_id": "trial-other"}},
    ],
)
async def test_unknown_queued_attribution_stays_queued_without_publishing(
    fifo, metadata
):
    db, publisher, current = fifo
    db.get_latest_user_message_in_session.return_value = pending(metadata)
    with pytest.raises(ValueError):
        await turn_queue.dispatch_next_for_user("user-1")
    db.update_chat_session_status.assert_any_await(
        session_id="session-1", expect_status="running", status="queued"
    )
    publisher.publish_message.assert_not_awaited()
    current.assert_not_awaited()
    assert get_trial_cost_context("user-1") is None
