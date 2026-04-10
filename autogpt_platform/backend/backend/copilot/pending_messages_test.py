"""Tests for the copilot pending-messages buffer.

Uses a fake async Redis client so the tests don't require a real Redis
instance (the backend test suite's DB/Redis fixtures are heavyweight
and pull in the full app startup).
"""

import json
from typing import Any

import pytest

from backend.copilot import pending_messages as pm_module
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    clear_pending_messages,
    drain_pending_messages,
    format_pending_as_user_message,
    peek_pending_count,
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

    async def eval(self, script: str, num_keys: int, *args: Any) -> Any:
        """Emulate the push Lua script.

        The real Lua script runs atomically in Redis; the fake
        implementation just runs the equivalent list operations in
        order and returns the final LLEN.  That's enough to exercise
        the cap + ordering invariants the tests care about.
        """
        key = args[0]
        payload = args[1]
        max_len = int(args[2])
        # ARGV[3] is TTL — fake doesn't enforce expiry
        lst = self.lists.setdefault(key, [])
        lst.append(payload)
        if len(lst) > max_len:
            # RPUSH + LTRIM(-N, -1) = keep only last N
            self.lists[key] = lst[-max_len:]
        return len(self.lists[key])

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
        context={"url": "https://example.com"},
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
