"""A wake for answered cards that the turn caps hold back is sent to PostHog."""

from contextlib import asynccontextmanager, nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import active_turns, turn_queue
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.gate import held
from backend.copilot.turn_queue import InflightCapExceeded


@pytest.fixture
def idle_chat(monkeypatch):
    """A chat with one answered card, idle and free of open questions."""
    call = SimpleNamespace(
        review_id="review-1",
        held_at=datetime(2026, 11, 10, tzinfo=UTC),
        envelope=MagicMock(),
        cost_context=SimpleNamespace(user_id="user-1"),
    )
    info = MagicMock(user_id="user-1", chat_status="idle", archived_at=None)
    info.metadata.pending_question = None
    info.metadata.delegated_by_session_id = None
    db = MagicMock(get_chat_session_metadata=AsyncMock(return_value=info))

    @asynccontextmanager
    async def full(*_args, **_kwargs):
        raise ConcurrentTurnLimitError("full", cap=5)
        yield

    monkeypatch.setattr(held, "_restore", AsyncMock())
    monkeypatch.setattr(held, "answered", AsyncMock(return_value=[call]))
    monkeypatch.setattr(held, "_waiting", AsyncMock(return_value=0))
    monkeypatch.setattr(held, "chat_db", lambda: db)
    monkeypatch.setattr(held, "restore_cost_context", lambda *_: nullcontext())
    monkeypatch.setattr(
        "backend.copilot.session_permissions.resolve_session_permissions",
        lambda _info: None,
    )
    monkeypatch.setattr(active_turns, "acquire_turn_slot", full)
    enqueue = AsyncMock()
    monkeypatch.setattr(turn_queue, "try_enqueue_turn", enqueue)
    return enqueue


@pytest.mark.asyncio
async def test_a_wake_queued_behind_the_running_cap_is_sent_as_queued(
    idle_chat, chat_blocked_events
):
    assert await held.wake("user-1", "sess-1") is True

    idle_chat.assert_awaited_once()
    assert [
        (e["limit"], e["outcome"], e["surface"], e["cap"]) for e in chat_blocked_events
    ] == [("concurrency", "queued", "background", 5)]


@pytest.mark.asyncio
async def test_a_wake_past_the_inflight_cap_is_sent_as_refused(
    idle_chat, chat_blocked_events
):
    idle_chat.side_effect = InflightCapExceeded(used=15, cap=15)

    assert await held.wake("user-1", "sess-1") is True

    assert [
        (e["limit"], e["outcome"], e["used"], e["cap"]) for e in chat_blocked_events
    ] == [("concurrency", "refused", 15, 15)]
