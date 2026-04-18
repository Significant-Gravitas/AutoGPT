"""Unit tests for pending_message_helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import pending_message_helpers as helpers_module
from backend.copilot.pending_message_helpers import (
    PENDING_CALL_LIMIT,
    check_pending_call_rate,
    combine_pending_with_current,
    drain_pending_safe,
    insert_pending_before_last,
    persist_session_safe,
)
from backend.copilot.pending_messages import PendingMessage

# ── check_pending_call_rate ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_pending_call_rate_returns_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module, "get_redis_async", AsyncMock(return_value=MagicMock())
    )
    monkeypatch.setattr(helpers_module, "incr_with_ttl", AsyncMock(return_value=3))

    result = await check_pending_call_rate("user-1")
    assert result == 3


@pytest.mark.asyncio
async def test_check_pending_call_rate_fails_open_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("down")),
    )

    result = await check_pending_call_rate("user-1")
    assert result == 0


@pytest.mark.asyncio
async def test_check_pending_call_rate_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module, "get_redis_async", AsyncMock(return_value=MagicMock())
    )
    monkeypatch.setattr(
        helpers_module,
        "incr_with_ttl",
        AsyncMock(return_value=PENDING_CALL_LIMIT + 1),
    )

    result = await check_pending_call_rate("user-1")
    assert result > PENDING_CALL_LIMIT


# ── drain_pending_safe ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_pending_safe_returns_pending_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``drain_pending_safe`` now returns the structured ``PendingMessage``
    objects (not pre-formatted strings) so the auto-continue re-queue path
    can preserve ``file_ids`` / ``context`` on rollback."""
    msgs = [
        PendingMessage(content="hello", file_ids=["f1"]),
        PendingMessage(content="world"),
    ]
    monkeypatch.setattr(
        helpers_module, "drain_pending_messages", AsyncMock(return_value=msgs)
    )

    result = await drain_pending_safe("sess-1")
    assert result == msgs
    # Structured metadata survives — the bug r3105523410 guard.
    assert result[0].file_ids == ["f1"]


@pytest.mark.asyncio
async def test_drain_pending_safe_returns_empty_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers_module,
        "drain_pending_messages",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )

    result = await drain_pending_safe("sess-1", "[Test]")
    assert result == []


@pytest.mark.asyncio
async def test_drain_pending_safe_empty_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        helpers_module, "drain_pending_messages", AsyncMock(return_value=[])
    )

    result = await drain_pending_safe("sess-1")
    assert result == []


# ── combine_pending_with_current ───────────────────────────────────────


def test_combine_before_current_when_pending_older() -> None:
    """Pending typed before the /stream request → goes ahead of current
    (prior-turn / inter-turn case)."""
    pending = [
        PendingMessage(content="older_a", enqueued_at=100.0),
        PendingMessage(content="older_b", enqueued_at=110.0),
    ]
    result = combine_pending_with_current(
        pending, "current_msg", request_arrival_at=120.0
    )
    assert result == "older_a\n\nolder_b\n\ncurrent_msg"


def test_combine_after_current_when_pending_newer() -> None:
    """Pending queued AFTER the /stream request arrived → goes after
    current.  This is the race path where user hits enter twice in quick
    succession (second press goes through the queue endpoint while the
    first /stream is still processing)."""
    pending = [
        PendingMessage(content="race_followup", enqueued_at=125.0),
    ]
    result = combine_pending_with_current(
        pending, "current_msg", request_arrival_at=120.0
    )
    assert result == "current_msg\n\nrace_followup"


def test_combine_mixed_before_and_after() -> None:
    """Mixed bucket: older items first, current, then newer race items."""
    pending = [
        PendingMessage(content="way_older", enqueued_at=50.0),
        PendingMessage(content="race_fast_follow", enqueued_at=125.0),
        PendingMessage(content="also_older", enqueued_at=80.0),
    ]
    result = combine_pending_with_current(
        pending, "current_msg", request_arrival_at=120.0
    )
    # Enqueue order preserved within each bucket (stable partition).
    assert result == "way_older\n\nalso_older\n\ncurrent_msg\n\nrace_fast_follow"


def test_combine_no_current_joins_pending() -> None:
    """Auto-continue case: no current message, just drained pending."""
    pending = [PendingMessage(content="a"), PendingMessage(content="b")]
    result = combine_pending_with_current(pending, None, request_arrival_at=0.0)
    assert result == "a\n\nb"


def test_combine_legacy_zero_timestamp_sorts_before() -> None:
    """A ``PendingMessage`` from before this field existed (default 0.0)
    should sort as "before everything" — safe pre-fix behaviour."""
    pending = [PendingMessage(content="legacy", enqueued_at=0.0)]
    result = combine_pending_with_current(
        pending, "current_msg", request_arrival_at=120.0
    )
    assert result == "legacy\n\ncurrent_msg"


def test_combine_missing_request_arrival_falls_back_to_before() -> None:
    """If the HTTP handler didn't stamp ``request_arrival_at`` (0.0
    default — older queue entries) the combine degrades gracefully to
    the pre-fix behaviour: all pending goes before current."""
    pending = [
        PendingMessage(content="a", enqueued_at=500.0),
        PendingMessage(content="b", enqueued_at=1000.0),
    ]
    result = combine_pending_with_current(pending, "current", request_arrival_at=0.0)
    assert result == "a\n\nb\n\ncurrent"


# ── insert_pending_before_last ─────────────────────────────────────────


def _make_session(*contents: str) -> Any:
    session = MagicMock()
    session.messages = [MagicMock(role="user", content=c) for c in contents]
    return session


def test_insert_pending_before_last_single_existing_message() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, ["queued"])
    assert session.messages[0].content == "queued"
    assert session.messages[1].content == "current"


def test_insert_pending_before_last_multiple_pending() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, ["p1", "p2"])
    contents = [m.content for m in session.messages]
    assert contents == ["p1", "p2", "current"]


def test_insert_pending_before_last_empty_session() -> None:
    session = _make_session()
    insert_pending_before_last(session, ["queued"])
    assert session.messages[0].content == "queued"


def test_insert_pending_before_last_no_texts_is_noop() -> None:
    session = _make_session("current")
    insert_pending_before_last(session, [])
    assert len(session.messages) == 1


# ── persist_session_safe ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_session_safe_returns_updated_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MagicMock()
    updated = MagicMock()
    monkeypatch.setattr(
        helpers_module, "upsert_chat_session", AsyncMock(return_value=updated)
    )

    result = await persist_session_safe(original, "[Test]")
    assert result is updated


@pytest.mark.asyncio
async def test_persist_session_safe_returns_original_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MagicMock()
    monkeypatch.setattr(
        helpers_module,
        "upsert_chat_session",
        AsyncMock(side_effect=Exception("db error")),
    )

    result = await persist_session_safe(original, "[Test]")
    assert result is original


# ── persist_pending_as_user_rows ───────────────────────────────────────


class _FakeTranscript:
    """Minimal TranscriptBuilder shim — records append_user + snapshot/restore."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def append_user(self, content: str, uuid: str | None = None) -> None:
        self.entries.append(content)

    def snapshot(self) -> list[str]:
        return list(self.entries)

    def restore(self, snap: list[str]) -> None:
        self.entries = list(snap)


def _make_chat_message_class(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Return a simple ChatMessage stand-in that tracks sequence."""

    class _Msg:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content
            self.sequence: int | None = None

    monkeypatch.setattr(helpers_module, "ChatMessage", _Msg)
    return _Msg


@pytest.mark.asyncio
async def test_persist_pending_empty_list_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.messages = []
    tb = _FakeTranscript()
    monkeypatch.setattr(helpers_module, "upsert_chat_session", AsyncMock())
    monkeypatch.setattr(helpers_module, "push_pending_message", AsyncMock())

    ok = await persist_pending_as_user_rows(session, tb, [], log_prefix="[T]")
    assert ok is True
    assert session.messages == []
    assert tb.entries == []


@pytest.mark.asyncio
async def test_persist_pending_happy_path_appends_and_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    session.messages = []
    tb = _FakeTranscript()

    async def _fake_upsert(sess: Any) -> Any:
        # Simulate the DB back-filling sequence numbers on success.
        for i, m in enumerate(sess.messages):
            m.sequence = i
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _fake_upsert)
    push_mock = AsyncMock()
    monkeypatch.setattr(helpers_module, "push_pending_message", push_mock)

    pending = [PM(content="a"), PM(content="b")]
    ok = await persist_pending_as_user_rows(session, tb, pending, log_prefix="[T]")
    assert ok is True
    assert [m.content for m in session.messages] == ["a", "b"]
    assert tb.entries == ["a", "b"]
    push_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_pending_rollback_when_sequence_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    # Prior state — anchor point is len(messages) before the helper runs.
    session.messages = []
    tb = _FakeTranscript()
    tb.entries = ["earlier-entry"]

    async def _fake_upsert_fails_silently(sess: Any) -> Any:
        # Simulate the "persist swallowed the error" branch — sequences stay None.
        return sess

    monkeypatch.setattr(
        helpers_module, "upsert_chat_session", _fake_upsert_fails_silently
    )
    push_mock = AsyncMock()
    monkeypatch.setattr(helpers_module, "push_pending_message", push_mock)

    pending = [PM(content="a"), PM(content="b")]
    ok = await persist_pending_as_user_rows(session, tb, pending, log_prefix="[T]")

    assert ok is False
    # Rollback: session.messages trimmed to anchor, transcript restored.
    assert session.messages == []
    assert tb.entries == ["earlier-entry"]
    # Both pending messages re-queued.
    assert push_mock.await_count == 2
    assert push_mock.await_args_list[0].args[1] is pending[0]
    assert push_mock.await_args_list[1].args[1] is pending[1]


@pytest.mark.asyncio
async def test_persist_pending_rollback_calls_on_rollback_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline's openai_messages trim runs via the on_rollback hook."""
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    session.messages = []
    tb = _FakeTranscript()

    async def _fails(sess: Any) -> Any:
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _fails)
    monkeypatch.setattr(helpers_module, "push_pending_message", AsyncMock())

    on_rollback_calls: list[int] = []

    def _on_rollback(anchor: int) -> None:
        on_rollback_calls.append(anchor)

    await persist_pending_as_user_rows(
        session,
        tb,
        [PM(content="x")],
        log_prefix="[T]",
        on_rollback=_on_rollback,
    )
    assert on_rollback_calls == [0]


@pytest.mark.asyncio
async def test_persist_pending_uses_custom_content_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    session.messages = []
    tb = _FakeTranscript()

    async def _ok(sess: Any) -> Any:
        for i, m in enumerate(sess.messages):
            m.sequence = i
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _ok)
    monkeypatch.setattr(helpers_module, "push_pending_message", AsyncMock())

    await persist_pending_as_user_rows(
        session,
        tb,
        [PM(content="raw")],
        log_prefix="[T]",
        content_of=lambda pm: f"FORMATTED:{pm.content}",
    )
    assert session.messages[0].content == "FORMATTED:raw"
    assert tb.entries == ["FORMATTED:raw"]


@pytest.mark.asyncio
async def test_persist_pending_swallows_requeue_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken push_pending_message on rollback must not raise upward —
    the rollback still needs to trim state even if re-queue fails."""
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    session.messages = []
    tb = _FakeTranscript()

    async def _fails(sess: Any) -> Any:
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _fails)
    monkeypatch.setattr(
        helpers_module,
        "push_pending_message",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )

    ok = await persist_pending_as_user_rows(
        session, tb, [PM(content="x")], log_prefix="[T]"
    )
    # Still returns False (rolled back) — exception was logged + swallowed.
    assert ok is False
