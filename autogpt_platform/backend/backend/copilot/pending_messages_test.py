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
    clear_pending_messages,
    drain_pending_messages,
    format_pending_as_user_message,
    peek_pending_count,
    peek_pending_messages,
    push_pending_message,
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
    await clear_pending_messages("sess4")
    assert await peek_pending_count("sess4") == 0


@pytest.mark.asyncio
async def test_clear_is_idempotent(fake_redis: _FakeRedis) -> None:
    # Clearing an already-empty buffer should not raise
    await clear_pending_messages("sess_empty")
    await clear_pending_messages("sess_empty")


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
