"""Unit tests for turn_queue: per-user FIFO queue layered over
ChatSession.chatStatus.

DB access is mocked via the ``backend.copilot.turn_queue.chat_db``
indirection — same accessor pattern the executor subprocess uses to
RPC into ``DatabaseManager``. Patching the accessor avoids reaching
for Prisma directly while still exercising the queue's branching.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import turn_queue
from backend.copilot.model import ChatMessage as PydanticChatMessage


class _NoopAsyncCM:
    """Stand-in for the Redis NX session lock context manager."""

    async def __aenter__(self):
        return True

    async def __aexit__(self, *exc):
        return None


def _pyd_message(**overrides) -> PydanticChatMessage:
    """Build a Pydantic ChatMessage with sensible defaults."""
    base = {
        "id": "msg-1",
        "role": "user",
        "content": "hello",
        "session_id": "s1",
        "metadata": None,
        "created_at": datetime.now(timezone.utc),
        "sequence": 1,
    }
    base.update(overrides)
    return PydanticChatMessage(**base)


def _mock_session(session_id: str = "s1", title: str | None = "T") -> MagicMock:
    """Build a ChatSessionInfo-ish mock for list_chat_sessions_by_status
    return values (the function returns app-model rows, not raw Prisma,
    so the RPC serializer can pass them through)."""
    s = MagicMock()
    s.session_id = session_id
    s.title = title
    s.updated_at = datetime.now(timezone.utc)
    return s


# ── enqueue_turn payload encoding ──────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_turn_packs_metadata_into_metadata_payload() -> None:
    """Non-message dispatch params (file_ids, mode, model, permissions,
    context, request_arrival_at) land in the ChatMessage row's
    ``metadata`` JSONB so the dispatcher can replay the original turn
    shape later."""
    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=42)
    db.add_chat_message = AsyncMock(return_value=_pyd_message(sequence=42))
    db.update_chat_session_status = AsyncMock(return_value=True)
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "_get_session_lock", return_value=_NoopAsyncCM()),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        await turn_queue.enqueue_turn(
            user_id="u1",
            session_id="s1",
            message="hello",
            message_id="msg-1",
            context={"url": "https://example.com"},
            file_ids=["f1", "f2"],
            mode="extended_thinking",
            model="advanced",
            permissions={"tool_filter": "allow"},
            request_arrival_at=123.45,
        )
    kwargs = db.add_chat_message.call_args.kwargs
    assert kwargs["session_id"] == "s1"
    assert kwargs["sequence"] == 42
    metadata = kwargs["metadata"]
    assert metadata["context"] == {"url": "https://example.com"}
    assert metadata["file_ids"] == ["f1", "f2"]
    assert metadata["mode"] == "extended_thinking"
    assert metadata["model"] == "advanced"
    assert metadata["permissions"] == {"tool_filter": "allow"}
    assert metadata["request_arrival_at"] == 123.45
    # Session is flipped idle → queued.
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="idle", status="queued", user_id="u1"
    )


@pytest.mark.asyncio
async def test_enqueue_turn_omits_null_fields_from_metadata() -> None:
    """A turn with no extra params leaves ``metadata`` NULL on the row
    instead of an empty object."""
    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=1)
    db.add_chat_message = AsyncMock(return_value=_pyd_message())
    db.update_chat_session_status = AsyncMock(return_value=True)
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "_get_session_lock", return_value=_NoopAsyncCM()),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        await turn_queue.enqueue_turn(user_id="u1", session_id="s1", message="hello")
    assert db.add_chat_message.call_args.kwargs["metadata"] is None


# ── cancel_queued_turn ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_queued_turn_returns_true_and_invalidates_cache() -> None:
    """A successful cancel flips the session ``queued`` → ``idle`` and
    invalidates the session cache so the frontend drops the badge."""
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    invalidate = AsyncMock()
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "invalidate_session_cache", new=invalidate),
    ):
        ok = await turn_queue.cancel_queued_turn(user_id="u1", session_id="s1")
    assert ok is True
    invalidate.assert_awaited_once_with("s1")
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1",
        expect_status="queued",
        status="idle",
        user_id="u1",
    )


@pytest.mark.asyncio
async def test_cancel_queued_turn_returns_false_when_not_owned_or_not_queued() -> None:
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=False)
    with patch.object(turn_queue, "chat_db", return_value=db):
        ok = await turn_queue.cancel_queued_turn(user_id="u1", session_id="s1")
    assert ok is False


# ── try_enqueue_turn ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_try_enqueue_turn_raises_when_at_inflight_cap() -> None:
    """Pre-check rejects when running + queued already equals the cap."""
    db = MagicMock()
    db.count_chat_sessions_by_status = AsyncMock(return_value=10)
    db.add_chat_message = AsyncMock()
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "count_running_turns", new=AsyncMock(return_value=5)),
    ):
        with pytest.raises(turn_queue.InflightCapExceeded):
            await turn_queue.try_enqueue_turn(
                user_id="u1",
                inflight_cap=15,
                session_id="s1",
                message="hi",
            )
    db.add_chat_message.assert_not_awaited()


# ── dispatch_next_for_user ─────────────────────────────────────────────


def _patch_queued_list(rows):
    """Patch ``list_queued_sessions`` (the dispatcher's queue read) to
    return the given rows.  Patching the helper rather than the
    underlying RPC keeps the test independent of how chat_db()
    resolves in-process vs. via DatabaseManagerAsyncClient."""
    return patch.object(
        turn_queue, "list_queued_sessions", new=AsyncMock(return_value=rows)
    )


@pytest.mark.asyncio
async def test_dispatch_returns_false_when_queue_empty() -> None:
    with _patch_queued_list([]):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False


@pytest.mark.asyncio
async def test_dispatch_leaves_queued_when_user_paywalled() -> None:
    """A queued head whose owner has lapsed to NO_TIER stays queued —
    no transition fires."""
    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_mock_session()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=True),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_leaves_queued_on_rate_limit_exceeded() -> None:
    """Mid-queue rate-limit lapse: leave the head queued, the next tick
    re-validates."""
    from backend.copilot.rate_limit import RateLimitExceeded

    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_mock_session()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(
                side_effect=RateLimitExceeded(
                    "daily", resets_at=datetime.now(timezone.utc)
                )
            ),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_defers_on_rate_limit_unavailable() -> None:
    from backend.copilot.rate_limit import RateLimitUnavailable

    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_mock_session()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(side_effect=RateLimitUnavailable()),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_happy_path_claims_and_dispatches() -> None:
    """All gates pass → claim session queued → running, build a TurnSlot,
    dispatch_turn, invalidate cache, return True."""
    head = _mock_session(session_id="s1")
    pending = _pyd_message(metadata={"mode": "extended_thinking"})
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    invalidate = AsyncMock()
    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=invalidate),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is True
    dispatch_turn_mock.assert_awaited_once()
    invalidate.assert_awaited_once_with("s1")
    # Single claim transition fired (no restore).
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="queued", status="running"
    )


@pytest.mark.asyncio
async def test_dispatch_rolls_claim_back_on_dispatch_failure() -> None:
    """If ``dispatch_turn`` raises after claim, restore the session
    ``running`` → ``queued`` so the next tick can retry."""
    head = _mock_session(session_id="s1")
    pending = _pyd_message(metadata={"mode": "extended_thinking"})
    db = MagicMock()
    # First call (claim) returns True; second call (restore) also True.
    db.update_chat_session_status = AsyncMock(side_effect=[True, True])
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock(side_effect=RuntimeError("RabbitMQ blip"))
    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="RabbitMQ blip"):
            await turn_queue.dispatch_next_for_user("u1")
    # Two transitions: claim then restore.  Redis-side meta cleanup is
    # ``dispatch_turn``'s responsibility (its own try/finally on the
    # ``committed`` flag), not the dispatcher's — see the
    # ``test_dispatch_turn_cleans_redis_on_enqueue_failure`` test in
    # ``executor/utils_test`` for that contract.
    assert db.update_chat_session_status.await_count == 2
    db.update_chat_session_status.assert_any_await(
        session_id="s1", expect_status="queued", status="running"
    )
    db.update_chat_session_status.assert_any_await(
        session_id="s1", expect_status="running", status="queued"
    )
