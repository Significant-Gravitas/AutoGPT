"""Tests for the copilot pending-messages buffer.

Uses a fake async Redis client so the tests don't require a real Redis
instance (the backend test suite's DB/Redis fixtures are heavyweight
and pull in the full app startup).
"""

import asyncio
import json
from typing import Any

import pytest

from backend.copilot import pending_messages as pm_module
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    PendingMessageContext,
    _clear_pending_messages_unsafe,
    drain_and_format_for_injection,
    drain_pending_for_persist,
    drain_pending_messages,
    format_pending_as_followup,
    format_pending_as_user_message,
    peek_pending_count,
    peek_pending_messages,
    push_pending_message,
    stash_pending_for_persist,
)

# ── Fake Redis ──────────────────────────────────────────────────────


class _FakeRedis:
    def __init__(self) -> None:
        # Values are ``str | bytes`` because real redis-py returns
        # bytes when ``decode_responses=False``; the drain path must
        # handle both and our tests exercise both.
        self.lists: dict[str, list[str | bytes]] = {}
        self.published: list[tuple[str, str]] = []

    async def rpush(self, key: str, *values: Any) -> int:
        lst = self.lists.setdefault(key, [])
        lst.extend(values)
        return len(lst)

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        lst = self.lists.get(key, [])
        # Redis LTRIM stop is inclusive; -1 means the last element.
        if stop == -1:
            self.lists[key] = lst[start:]
        else:
            self.lists[key] = lst[start : stop + 1]

    async def expire(self, key: str, seconds: int) -> int:
        # Fake doesn't enforce TTL — just acknowledge.
        return 1

    async def publish(self, channel: str, payload: str) -> int:
        self.published.append((channel, payload))
        return 1

    async def lpop(self, key: str, count: int) -> list[str | bytes] | None:
        lst = self.lists.get(key)
        if not lst:
            return None
        popped = lst[:count]
        self.lists[key] = lst[count:]
        return popped

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def lrange(self, key: str, start: int, stop: int) -> list[str | bytes]:
        lst = self.lists.get(key, [])
        # Redis LRANGE stop is inclusive; -1 means the last element.
        if stop == -1:
            return list(lst[start:])
        return list(lst[start : stop + 1])

    async def delete(self, key: str) -> int:
        if key in self.lists:
            del self.lists[key]
            return 1
        return 0

    def pipeline(self, transaction: bool = True) -> "_FakePipeline":
        # Returns a fake pipeline that records ops and replays them in
        # order on ``execute()``.  Used by ``capped_rpush`` (push_pending_message)
        # and ``incr_with_ttl`` (call-rate check) via MULTI/EXEC.
        return _FakePipeline(self)

    async def incr(self, key: str) -> int:
        # Used by incr_with_ttl's pipeline.
        current = int(self.lists.get(key, [0])[0]) if self.lists.get(key) else 0
        current += 1
        # We abuse the same lists dict for simple counters — store [count].
        self.lists[key] = [str(current)]
        return current


class _FakePipeline:
    """Async pipeline shim matching the redis-py MULTI/EXEC surface."""

    def __init__(self, parent: "_FakeRedis") -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    # Each method just records the op; dispatching happens in execute().
    def rpush(self, key: str, *values: Any) -> "_FakePipeline":
        self._ops.append(("rpush", (key, *values), {}))
        return self

    def ltrim(self, key: str, start: int, stop: int) -> "_FakePipeline":
        self._ops.append(("ltrim", (key, start, stop), {}))
        return self

    def expire(self, key: str, seconds: int, **kw: Any) -> "_FakePipeline":
        self._ops.append(("expire", (key, seconds), kw))
        return self

    def llen(self, key: str) -> "_FakePipeline":
        self._ops.append(("llen", (key,), {}))
        return self

    def incr(self, key: str) -> "_FakePipeline":
        self._ops.append(("incr", (key,), {}))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for name, args, _kw in self._ops:
            fn = getattr(self._parent, name)
            results.append(await fn(*args))
        return results

    # Support `async with pipeline() as pipe:` too.
    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    redis = _FakeRedis()

    async def _get_redis_async() -> _FakeRedis:
        return redis

    monkeypatch.setattr(pm_module, "get_redis_async", _get_redis_async)
    return redis


# ── Basic push / drain ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_and_drain_single_message(fake_redis: _FakeRedis) -> None:
    length = await push_pending_message("sess1", PendingMessage(content="hello"))
    assert length == 1
    assert await peek_pending_count("sess1") == 1

    drained = await drain_pending_messages("sess1")
    assert len(drained) == 1
    assert drained[0].content == "hello"
    assert await peek_pending_count("sess1") == 0


@pytest.mark.asyncio
async def test_push_and_drain_preserves_order(fake_redis: _FakeRedis) -> None:
    for i in range(3):
        await push_pending_message("sess2", PendingMessage(content=f"msg {i}"))

    drained = await drain_pending_messages("sess2")
    assert [m.content for m in drained] == ["msg 0", "msg 1", "msg 2"]


@pytest.mark.asyncio
async def test_drain_empty_returns_empty_list(fake_redis: _FakeRedis) -> None:
    assert await drain_pending_messages("nope") == []


# ── Buffer cap ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cap_drops_oldest_when_exceeded(fake_redis: _FakeRedis) -> None:
    # Push MAX_PENDING_MESSAGES + 3 messages
    for i in range(MAX_PENDING_MESSAGES + 3):
        await push_pending_message("sess3", PendingMessage(content=f"m{i}"))

    # Buffer should be clamped to MAX
    assert await peek_pending_count("sess3") == MAX_PENDING_MESSAGES

    drained = await drain_pending_messages("sess3")
    assert len(drained) == MAX_PENDING_MESSAGES
    # Oldest 3 dropped — we should only see m3..m(MAX+2)
    assert drained[0].content == "m3"
    assert drained[-1].content == f"m{MAX_PENDING_MESSAGES + 2}"


# ── Clear ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_removes_buffer(fake_redis: _FakeRedis) -> None:
    await push_pending_message("sess4", PendingMessage(content="x"))
    await push_pending_message("sess4", PendingMessage(content="y"))
    await _clear_pending_messages_unsafe("sess4")
    assert await peek_pending_count("sess4") == 0


@pytest.mark.asyncio
async def test_clear_is_idempotent(fake_redis: _FakeRedis) -> None:
    # Clearing an already-empty buffer should not raise
    await _clear_pending_messages_unsafe("sess_empty")
    await _clear_pending_messages_unsafe("sess_empty")


# ── Publish hook ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_publishes_notification(fake_redis: _FakeRedis) -> None:
    await push_pending_message("sess5", PendingMessage(content="hi"))
    assert ("copilot:pending:notify:sess5", "1") in fake_redis.published


# ── Format helper ───────────────────────────────────────────────────


def test_format_pending_plain_text() -> None:
    msg = PendingMessage(content="just text")
    out = format_pending_as_user_message(msg)
    assert out == {"role": "user", "content": "just text"}


def test_format_pending_with_context_url() -> None:
    msg = PendingMessage(
        content="see this page",
        context=PendingMessageContext(url="https://example.com"),
    )
    out = format_pending_as_user_message(msg)
    content = out["content"]
    assert out["role"] == "user"
    assert "see this page" in content
    # The URL should appear verbatim in the [Page URL: ...] block.
    assert "[Page URL: https://example.com]" in content


def test_format_pending_with_file_ids() -> None:
    msg = PendingMessage(content="look here", file_ids=["a", "b"])
    out = format_pending_as_user_message(msg)
    assert "file_id=a" in out["content"]
    assert "file_id=b" in out["content"]


def test_format_pending_with_all_fields() -> None:
    """All fields (content + context url/content + file_ids) should all appear."""
    msg = PendingMessage(
        content="summarise this",
        context=PendingMessageContext(
            url="https://example.com/page",
            content="headline text",
        ),
        file_ids=["f1", "f2"],
    )
    out = format_pending_as_user_message(msg)
    body = out["content"]
    assert out["role"] == "user"
    assert "summarise this" in body
    assert "[Page URL: https://example.com/page]" in body
    assert "[Page content]\nheadline text" in body
    assert "file_id=f1" in body
    assert "file_id=f2" in body


# ── Followup block caps ────────────────────────────────────────────


def test_format_followup_single_message() -> None:
    out = format_pending_as_followup([PendingMessage(content="hello")])
    assert "<user_follow_up>" in out
    assert "</user_follow_up>" in out
    assert "Message 1:\nhello" in out


def test_format_followup_total_cap_drops_overflow() -> None:
    """10 × 2 KB messages must truncate past the total cap (~6 KB) with a
    marker indicating how many were dropped."""
    messages = [PendingMessage(content="A" * 2_000) for _ in range(10)]
    out = format_pending_as_followup(messages)
    # Block stays within the total cap (plus a little wrapper overhead).
    # The body alone is capped at 6 KB; we allow generous overhead for the
    # <user_follow_up> wrapper + headers.
    assert len(out) < 8_000
    assert "more message(s) truncated" in out
    # The first message at least must be present.
    assert "Message 1:" in out


def test_format_followup_total_cap_marker_counts_dropped() -> None:
    """The marker should name the exact number of dropped messages."""
    # Each 3 KB message gets capped to 2 KB first; with ~2 KB per entry and a
    # 6 KB total cap, roughly two entries fit and the rest are dropped.
    messages = [PendingMessage(content="X" * 3_000) for _ in range(5)]
    out = format_pending_as_followup(messages)
    assert "Message 1:" in out
    assert "Message 2:" in out
    # Message 3 would push total past 6 KB; marker should report exactly how
    # many were left out (here: messages 3, 4, 5 → 3 dropped).
    assert "[3 more message(s) truncated]" in out


def test_format_followup_empty_returns_empty_string() -> None:
    assert format_pending_as_followup([]) == ""


# ── Malformed payload handling ──────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_skips_malformed_entries(
    fake_redis: _FakeRedis,
) -> None:
    # Seed the fake with a mix of valid and malformed payloads
    fake_redis.lists["copilot:pending:bad"] = [
        json.dumps({"content": "valid"}),
        "{not valid json",
        json.dumps({"content": "also valid", "file_ids": ["a"]}),
    ]
    drained = await drain_pending_messages("bad")
    assert len(drained) == 2
    assert drained[0].content == "valid"
    assert drained[1].content == "also valid"


@pytest.mark.asyncio
async def test_drain_decodes_bytes_payloads(
    fake_redis: _FakeRedis,
) -> None:
    """Real redis-py returns ``bytes`` when ``decode_responses=False``.

    Seed the fake with bytes values to exercise the ``decode("utf-8")``
    branch in ``drain_pending_messages`` so a regression there doesn't
    slip past CI.
    """
    fake_redis.lists["copilot:pending:bytes_sess"] = [
        json.dumps({"content": "from bytes"}).encode("utf-8"),
    ]
    drained = await drain_pending_messages("bytes_sess")
    assert len(drained) == 1
    assert drained[0].content == "from bytes"


@pytest.mark.asyncio
async def test_peek_decodes_bytes_payloads(
    fake_redis: _FakeRedis,
) -> None:
    """``peek_pending_messages`` uses the same ``_decode_redis_item`` helper
    as the drain path.  Seed with bytes to guard against regression.
    """
    fake_redis.lists["copilot:pending:peek_bytes_sess"] = [
        json.dumps({"content": "peeked from bytes"}).encode("utf-8"),
    ]
    peeked = await peek_pending_messages("peek_bytes_sess")
    assert len(peeked) == 1
    assert peeked[0].content == "peeked from bytes"
    # peek must NOT consume the item
    assert fake_redis.lists["copilot:pending:peek_bytes_sess"] != []


# ── Concurrency ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_push_and_drain(fake_redis: _FakeRedis) -> None:
    """Two pushes fired concurrently should both land; a concurrent drain
    should see at least one of them (the fake serialises, so it will
    always see both, but we exercise the code path either way)."""
    await asyncio.gather(
        push_pending_message("sess_conc", PendingMessage(content="a")),
        push_pending_message("sess_conc", PendingMessage(content="b")),
    )
    drained = await drain_pending_messages("sess_conc")
    assert len(drained) >= 1
    contents = {m.content for m in drained}
    assert contents <= {"a", "b"}


# ── Publish error path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_survives_publish_failure(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish error must not propagate — the buffer is still authoritative."""

    async def _fail_publish(channel: str, payload: str) -> int:
        raise RuntimeError("redis publish down")

    monkeypatch.setattr(fake_redis, "publish", _fail_publish)

    length = await push_pending_message("sess_pub_err", PendingMessage(content="ok"))
    assert length == 1
    drained = await drain_pending_messages("sess_pub_err")
    assert len(drained) == 1
    assert drained[0].content == "ok"


# ── peek_pending_messages ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_peek_pending_messages_returns_all_without_consuming(
    fake_redis: _FakeRedis,
) -> None:
    """Peek returns all queued messages and leaves the buffer intact."""
    await push_pending_message("peek1", PendingMessage(content="first"))
    await push_pending_message("peek1", PendingMessage(content="second"))

    peeked = await peek_pending_messages("peek1")
    assert len(peeked) == 2
    assert peeked[0].content == "first"
    assert peeked[1].content == "second"

    # Buffer must not be consumed — count still 2
    assert await peek_pending_count("peek1") == 2
    drained = await drain_pending_messages("peek1")
    assert len(drained) == 2


@pytest.mark.asyncio
async def test_peek_pending_messages_empty_buffer(fake_redis: _FakeRedis) -> None:
    """Peek on a missing key returns an empty list without raising."""
    result = await peek_pending_messages("no_such_session")
    assert result == []


@pytest.mark.asyncio
async def test_peek_pending_messages_decodes_bytes_payloads(
    fake_redis: _FakeRedis,
) -> None:
    """peek_pending_messages decodes bytes entries the same way drain does."""
    fake_redis.lists["copilot:pending:peek_bytes"] = [
        json.dumps({"content": "from bytes"}).encode("utf-8"),
    ]
    peeked = await peek_pending_messages("peek_bytes")
    assert len(peeked) == 1
    assert peeked[0].content == "from bytes"


@pytest.mark.asyncio
async def test_peek_pending_messages_skips_malformed_entries(
    fake_redis: _FakeRedis,
) -> None:
    """Malformed entries are skipped and valid ones are returned."""
    fake_redis.lists["copilot:pending:peek_bad"] = [
        json.dumps({"content": "valid peek"}),
        "{bad json",
        json.dumps({"content": "also valid peek"}),
    ]
    peeked = await peek_pending_messages("peek_bad")
    assert len(peeked) == 2
    assert peeked[0].content == "valid peek"
    assert peeked[1].content == "also valid peek"


# ── Persist queue (mid-turn follow-up UI bubble hand-off) ───────────


@pytest.mark.asyncio
async def test_stash_for_persist_enqueues_and_drain_pops_in_order(
    fake_redis: _FakeRedis,
) -> None:
    """stash_pending_for_persist writes messages under the persist key;
    drain_pending_for_persist LPOPs them in enqueue order."""
    msgs = [
        PendingMessage(content="first mid-turn follow-up"),
        PendingMessage(content="second"),
    ]
    await stash_pending_for_persist("sess-persist", msgs)

    # Stored under the distinct persist key, NOT the primary buffer.
    assert "copilot:pending-persist:sess-persist" in fake_redis.lists
    assert "copilot:pending:sess-persist" not in fake_redis.lists

    drained = await drain_pending_for_persist("sess-persist")
    assert len(drained) == 2
    assert drained[0].content == "first mid-turn follow-up"
    assert drained[1].content == "second"

    # Queue is empty after drain.
    assert await drain_pending_for_persist("sess-persist") == []


@pytest.mark.asyncio
async def test_stash_for_persist_empty_list_is_noop(
    fake_redis: _FakeRedis,
) -> None:
    """Passing an empty list must NOT create a Redis key (would leak
    empty persist entries and require a drain for no reason)."""
    await stash_pending_for_persist("sess-noop", [])
    assert "copilot:pending-persist:sess-noop" not in fake_redis.lists


@pytest.mark.asyncio
async def test_drain_pending_for_persist_missing_key_returns_empty(
    fake_redis: _FakeRedis,
) -> None:
    assert await drain_pending_for_persist("never-stashed") == []


@pytest.mark.asyncio
async def test_drain_pending_for_persist_skips_malformed(
    fake_redis: _FakeRedis,
) -> None:
    fake_redis.lists["copilot:pending-persist:bad"] = [
        json.dumps({"content": "good one"}),
        "not json",
        json.dumps({"content": "another good one"}),
    ]
    result = await drain_pending_for_persist("bad")
    assert [m.content for m in result] == ["good one", "another good one"]


@pytest.mark.asyncio
async def test_persist_queue_isolated_from_primary_buffer(
    fake_redis: _FakeRedis,
) -> None:
    """Draining the persist queue must NOT touch the primary pending
    buffer (and vice versa) — they serve different lifecycles."""
    # Seed the primary buffer with one entry.
    await push_pending_message("sess-iso", PendingMessage(content="primary"))
    # Stash a separate entry on the persist queue.
    await stash_pending_for_persist("sess-iso", [PendingMessage(content="persist")])

    drained_persist = await drain_pending_for_persist("sess-iso")
    assert [m.content for m in drained_persist] == ["persist"]

    # Primary buffer untouched.
    assert await peek_pending_count("sess-iso") == 1
    drained_primary = await drain_pending_messages("sess-iso")
    assert [m.content for m in drained_primary] == ["primary"]


@pytest.mark.asyncio
async def test_stash_for_persist_swallows_redis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken Redis during stash must not raise — Claude has already
    seen the follow-up via tool output; the only fallout is a missing
    UI bubble, which we log and move on."""

    async def _broken_redis() -> Any:
        raise ConnectionError("redis down")

    monkeypatch.setattr(pm_module, "get_redis_async", _broken_redis)

    # Must NOT raise.
    await stash_pending_for_persist("sess-broken", [PendingMessage(content="lost")])


# ── drain_and_format_for_injection: shared entry point ─────────────────


@pytest.mark.asyncio
async def test_drain_and_format_for_injection_happy_path(
    fake_redis: _FakeRedis,
) -> None:
    """Queued messages drain into a ready-to-inject <user_follow_up> block
    AND are stashed on the persist queue for UI row hand-off."""
    await push_pending_message("sess-share", PendingMessage(content="do X also"))

    result = await drain_and_format_for_injection("sess-share", log_prefix="[TEST]")

    assert "<user_follow_up>" in result
    assert "do X also" in result
    # Primary buffer drained.
    assert await peek_pending_count("sess-share") == 0
    # Persist queue got a copy for the UI.
    persisted = await drain_pending_for_persist("sess-share")
    assert len(persisted) == 1
    assert persisted[0].content == "do X also"


@pytest.mark.asyncio
async def test_drain_and_format_for_injection_empty_returns_empty(
    fake_redis: _FakeRedis,
) -> None:
    assert await drain_and_format_for_injection("sess-empty", log_prefix="[TEST]") == ""


@pytest.mark.asyncio
async def test_drain_and_format_for_injection_swallows_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _broken() -> Any:
        raise ConnectionError("down")

    monkeypatch.setattr(pm_module, "get_redis_async", _broken)

    # Must NOT raise — broken Redis becomes "nothing to inject".
    assert (
        await drain_and_format_for_injection("sess-broken", log_prefix="[TEST]") == ""
    )


@pytest.mark.asyncio
async def test_drain_and_format_for_injection_missing_session_id() -> None:
    assert await drain_and_format_for_injection("", log_prefix="[TEST]") == ""
