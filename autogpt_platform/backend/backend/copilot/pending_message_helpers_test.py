"""Unit tests for pending_message_helpers."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.copilot import pending_message_helpers as helpers_module
from backend.copilot.pending_message_helpers import (
    PENDING_CALL_LIMIT,
    QueuePendingMessageResponse,
    StreamRegistryUnavailable,
    check_pending_call_rate,
    combine_pending_with_current,
    drain_pending_safe,
    insert_pending_before_last,
    is_turn_in_flight,
    persist_session_safe,
    queue_pending_for_http,
)
from backend.copilot.pending_messages import MAX_PENDING_MESSAGES, PendingMessage

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


# ── is_turn_in_flight: fail-closed on Redis errors ────────────────────


def _mock_chat_db(
    monkeypatch: pytest.MonkeyPatch, *, status: str | None = "idle"
) -> None:
    """Stub ``chat_db()`` so the ``is_turn_in_flight`` fallthrough that
    reads ``ChatSession.chatStatus`` doesn't hit a real DB connection."""
    db = MagicMock()
    db.get_chat_session_status = AsyncMock(return_value=status)
    monkeypatch.setattr(helpers_module, "chat_db", lambda: db)


@pytest.mark.asyncio
async def test_is_turn_in_flight_returns_false_when_no_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis says no active stream AND ChatSession is idle → not in flight."""
    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(return_value=None),
    )
    _mock_chat_db(monkeypatch, status="idle")
    assert await is_turn_in_flight("sess-1") is False


@pytest.mark.asyncio
async def test_is_turn_in_flight_returns_true_when_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued sessions are also in flight even though the Redis stream
    registry has no entry — the dispatcher hasn't claimed the row yet."""
    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(return_value=None),
    )
    _mock_chat_db(monkeypatch, status="queued")
    assert await is_turn_in_flight("sess-1") is True


@pytest.mark.asyncio
async def test_is_turn_in_flight_returns_true_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = MagicMock()
    active.status = "running"
    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(return_value=active),
    )
    _mock_chat_db(monkeypatch, status="running")
    assert await is_turn_in_flight("sess-1") is True


@pytest.mark.asyncio
async def test_is_turn_in_flight_raises_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis brown-out must NOT bubble as an unhandled 500. The helper
    raises a typed ``StreamRegistryUnavailable`` so the chat-route
    pre-flight chain can map it to 503 + Retry-After (matching the
    ``RateLimitUnavailable`` mapping for the next pre-flight step)."""
    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(side_effect=ConnectionError("redis down")),
    )
    with pytest.raises(StreamRegistryUnavailable):
        await is_turn_in_flight("sess-1")


@pytest.mark.asyncio
async def test_is_turn_in_flight_raises_on_redis_cluster_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RedisClusterException`` (e.g. ``SlotNotCoveredError`` during a
    GKE rolling restart) does NOT inherit from ``RedisError`` — verify it
    is caught explicitly by the same fail-closed branch."""
    from redis.exceptions import RedisClusterException

    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(side_effect=RedisClusterException("slot not covered")),
    )
    with pytest.raises(StreamRegistryUnavailable):
        await is_turn_in_flight("sess-1")


@pytest.mark.asyncio
async def test_is_turn_in_flight_raises_when_chat_status_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB-side ``get_chat_session_status`` failure must surface as the
    same typed ``StreamRegistryUnavailable`` the Redis branch raises so
    the HTTP layer's existing 503 + Retry-After mapping covers it.
    Without this the Prisma error would bubble as a raw 500."""
    monkeypatch.setattr(
        helpers_module,
        "get_active_session_meta",
        AsyncMock(return_value=None),
    )
    db = MagicMock()
    db.get_chat_session_status = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(helpers_module, "chat_db", lambda: db)

    with pytest.raises(StreamRegistryUnavailable):
        await is_turn_in_flight("sess-1")


# ── queue_pending_for_http: gate-then-bump ordering ───────────────────


@pytest.mark.asyncio
async def test_queue_pending_does_not_charge_rate_on_toctou_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the Lua gate refuses the push because the turn just completed,
    the per-user call-rate counter must NOT have been incremented — bumping
    it before the gate would charge a budget tick for every TOCTOU loss
    against turn completion (race that both this endpoint and the POST
    /stream queue-fall-through can trigger)."""
    monkeypatch.setattr(
        helpers_module,
        "queue_user_message",
        AsyncMock(
            return_value=QueuePendingMessageResponse(
                buffer_length=0,
                max_buffer_length=MAX_PENDING_MESSAGES,
                turn_in_flight=False,
            )
        ),
    )
    rate_mock = AsyncMock(return_value=1)
    monkeypatch.setattr(helpers_module, "check_pending_call_rate", rate_mock)
    monkeypatch.setattr(
        helpers_module, "resolve_workspace_files", AsyncMock(return_value=[])
    )

    with pytest.raises(HTTPException) as exc_info:
        await queue_pending_for_http(
            session_id="sess-1",
            user_id="user-1",
            message="hi",
            context=None,
            file_ids=None,
        )
    assert exc_info.value.status_code == 409
    rate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_pending_charges_rate_only_after_successful_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = QueuePendingMessageResponse(
        buffer_length=2,
        max_buffer_length=MAX_PENDING_MESSAGES,
        turn_in_flight=True,
    )
    queue_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(helpers_module, "queue_user_message", queue_mock)
    rate_mock = AsyncMock(return_value=PENDING_CALL_LIMIT)
    monkeypatch.setattr(helpers_module, "check_pending_call_rate", rate_mock)
    monkeypatch.setattr(
        helpers_module, "resolve_workspace_files", AsyncMock(return_value=[])
    )

    result = await queue_pending_for_http(
        session_id="sess-1",
        user_id="user-1",
        message="hi",
        context=None,
        file_ids=None,
    )

    assert result is response
    queue_mock.assert_awaited_once()
    rate_mock.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_queue_pending_429_after_push_when_limit_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the post-push rate counter crosses the limit, the message stays
    in the buffer (next drain will pick it up) but the response is 429 so
    the client backs off."""
    response = QueuePendingMessageResponse(
        buffer_length=3,
        max_buffer_length=MAX_PENDING_MESSAGES,
        turn_in_flight=True,
    )
    queue_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(helpers_module, "queue_user_message", queue_mock)
    monkeypatch.setattr(
        helpers_module,
        "check_pending_call_rate",
        AsyncMock(return_value=PENDING_CALL_LIMIT + 1),
    )
    monkeypatch.setattr(
        helpers_module, "resolve_workspace_files", AsyncMock(return_value=[])
    )

    with pytest.raises(HTTPException) as exc_info:
        await queue_pending_for_http(
            session_id="sess-1",
            user_id="user-1",
            message="hi",
            context=None,
            file_ids=None,
        )
    assert exc_info.value.status_code == 429
    queue_mock.assert_awaited_once()


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
async def test_persist_pending_no_transcript_path_skips_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn-start drain: passing ``transcript_builder=None`` MUST NOT touch
    the transcript — it would triple-count pending entries in the next
    turn's ``--resume`` context (the combined ``current_message`` is what
    gets written to transcript at turn-end)."""
    from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    session.messages = []

    async def _fake_upsert(sess: Any) -> Any:
        for i, m in enumerate(sess.messages):
            m.sequence = i
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _fake_upsert)

    pending = [PM(content="chip-1"), PM(content="chip-2")]
    ok = await persist_pending_as_user_rows(session, None, pending, log_prefix="[T]")
    assert ok is True
    assert [m.content for m in session.messages] == ["chip-1", "chip-2"]
    assert [m.sequence for m in session.messages] == [0, 1]


@pytest.mark.asyncio
async def test_turn_start_drain_invariants_one_bubble_per_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locks the chip-queue invariant that's broken five different ways
    in this PR's history.

    After a turn-start drain in the SDK / baseline service, the on-disk
    ``session.messages`` MUST satisfy all of:

      1. **N+1 user rows total** (1 routes.py-saved row + N pending).
      2. **Original row carries the wrapped envelopes** but NOT any
         pending text — the bubble in the UI shows just the original
         send (markdown strips the trusted ``<memory_context>`` /
         ``<user_context>`` blocks for display).
      3. **Each pending row carries its raw chip text alone** — no
         envelopes, no neighbours' texts joined with ``\\n\\n``.
      4. The COMBINED ``current_message`` is what the model sees this
         turn (and lands in the transcript at turn-end via
         ``append_user``) — separate from the per-row DB state.

    Past regressions this test would have caught:

      a. Persisting pending BEFORE ``inject_user_context`` (pending row
         received the full wrapped+combined string from inject's
         reverse-walk → bubble showed everything twice).
      b. Wrapping the COMBINED text in ``inject_user_context`` (original
         row carried both texts → chip text duplicated across two
         bubbles).
      c. Pre-PR behaviour: combining everything into the existing row
         (one bubble with ``"send\\n\\nchip"`` joined, chip's
         cardinality lost).
    """
    from backend.copilot.pending_message_helpers import (
        combine_pending_with_current,
        persist_pending_as_user_rows,
    )
    from backend.copilot.pending_messages import PendingMessage as PM

    _make_chat_message_class(monkeypatch)
    session = MagicMock()
    session.session_id = "sess"
    # Routes.py pre-saved the original send at seq=0.  inject_user_context
    # then wrapped its content with envelopes (simulated here).
    session.messages = [
        MagicMock(
            role="user",
            content=(
                "<memory_context>\nfacts\n</memory_context>\n\n"
                "<user_context>\nbio\n</user_context>\n\n"
                "can you sleep for 2 seconds then 3 seconds"
            ),
            sequence=0,
        )
    ]

    async def _fake_upsert(sess: Any) -> Any:
        # Simulate DB sequence back-fill on the rows that don't have one.
        next_seq = (
            max(
                (m.sequence for m in sess.messages if m.sequence is not None),
                default=-1,
            )
            + 1
        )
        for m in sess.messages:
            if m.sequence is None:
                m.sequence = next_seq
                next_seq += 1
        return sess

    monkeypatch.setattr(helpers_module, "upsert_chat_session", _fake_upsert)

    pending = [PM(content="oh sleep 4 secs in between", enqueued_at=0.0)]
    original = "can you sleep for 2 seconds then 3 seconds"

    # Step 1: combine for the model's prompt.  Result is what the SDK CLI
    # sees as the current-turn user input.
    combined = combine_pending_with_current(pending, original, request_arrival_at=0.0)
    assert "can you sleep for 2 seconds then 3 seconds" in combined
    assert "oh sleep 4 secs in between" in combined

    # Step 2: persist pending as separate user rows; transcript untouched.
    ok = await persist_pending_as_user_rows(session, None, pending, log_prefix="[T]")
    assert ok is True

    # Invariant 1: N+1 user rows.
    assert len(session.messages) == 2

    # Invariant 2: original row keeps wrapped envelopes; no pending text.
    original_row = session.messages[0]
    assert "<memory_context>" in original_row.content
    assert "<user_context>" in original_row.content
    assert "can you sleep" in original_row.content
    assert "oh sleep 4 secs" not in original_row.content, (
        "regression: original row absorbed pending text — "
        "inject ran on the combined string instead of the original send"
    )

    # Invariant 3: pending row is raw chip text only.
    chip_row = session.messages[1]
    assert (
        chip_row.content == "oh sleep 4 secs in between"
    ), f"regression: chip row content drifted from raw text — got {chip_row.content!r}"
    assert "<memory_context>" not in chip_row.content
    assert "can you sleep" not in chip_row.content, (
        "regression: chip row absorbed original text — "
        "persist ran before inject, or the chip was joined with `\\n\\n`"
    )


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
