from unittest.mock import AsyncMock, call

import pytest

from backend.copilot import pending_message_helpers as helpers
from backend.copilot import pending_messages
from backend.copilot.pending_messages import PendingMessage


@pytest.mark.parametrize("first_batch", [[], [PendingMessage(content="valid")]])
async def test_backlog_budget_counts_raw_entries(
    monkeypatch: pytest.MonkeyPatch, first_batch: list[PendingMessage]
) -> None:
    tail = [PendingMessage(content="tail")]
    pop = AsyncMock(side_effect=[first_batch, tail])
    monkeypatch.setattr(helpers, "peek_pending_count", AsyncMock(return_value=13))
    monkeypatch.setattr(helpers, "drain_pending_messages", pop)

    assert await helpers.drain_pending_safe("session", drain_backlog=True) == (
        first_batch + tail
    )
    assert pop.await_args_list == [
        call("session", limit=10),
        call("session", limit=3),
    ]


async def test_backlog_retains_messages_popped_before_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = [PendingMessage(content="already popped")]
    monkeypatch.setattr(helpers, "peek_pending_count", AsyncMock(return_value=13))
    monkeypatch.setattr(
        helpers, "drain_pending_messages", AsyncMock(side_effect=[batch, OSError()])
    )
    assert await helpers.drain_pending_safe("session", drain_backlog=True) == batch


@pytest.mark.parametrize("limit,expected", [(3, 3), (20, 10)])
async def test_single_pop_respects_bounded_limit(
    monkeypatch: pytest.MonkeyPatch, limit: int, expected: int
) -> None:
    redis = AsyncMock()
    redis.lpop.return_value = []
    monkeypatch.setattr(
        pending_messages, "get_redis_async", AsyncMock(return_value=redis)
    )
    assert await pending_messages.drain_pending_messages("session", limit=limit) == []
    redis.lpop.assert_awaited_once_with("copilot:pending:{session}", expected)
