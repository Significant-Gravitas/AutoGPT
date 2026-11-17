"""A connection's reply budget, run as the relay's own Lua on fakeredis.

Each reply stream has a cap of its own, and a terminal answer frees its
request's slot, so a modified companion that answers after its workers stop
waiting can fill stream after stream that nobody reads again. These tests send
that traffic through the relay's router and check that what the connection's
reply streams hold stops at one budget, and comes back when a stream is read or
expires unread.
"""

import asyncio
import json
from contextlib import aclosing
from typing import Any

import pytest
from fakeredis.aioredis import FakeRedis

from backend.copilot.tools import local_pc_relay_backlog as relay_backlog
from backend.copilot.tools import local_pc_relay_routing as relay_routing
from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_backlog import STREAM_TTL_SECONDS
from backend.copilot.tools.local_pc_relay_protocol import (
    RelayBacklogExceeded,
    response_stream_key,
)
from backend.copilot.tools.local_pc_relay_routing import RelayResponseRouter
from backend.copilot.tools.local_pc_relay_transport import RedisRelayTransport

# The relay's limits for a frame, a reply stream and a connection (16, 64 and
# 256 MiB), scaled down so a test moves kilobytes.
FRAME = 1024
STREAM_LIMIT = 4 * FRAME
CONNECTION_LIMIT = 16 * FRAME
SESSION = "session-1"


def _file_contents(request_id: str, size: int = FRAME) -> str:
    """The shim's answer to ``request_id``, exactly ``size`` bytes long."""

    def envelope(content: str) -> str:
        return json.dumps(
            {"type": "FILE_CONTENTS", "id": request_id, "payload": {"content": content}}
        )

    raw = envelope("x" * (size - len(envelope("").encode())))
    assert len(raw.encode()) == size
    return raw


async def _answer_late(
    router: RelayResponseRouter, reply_id: str, request_id: str, size: int = FRAME
) -> None:
    """A worker sends a request, and the shim's answer comes in after it."""
    router.register_request(
        {"type": "FILE_READ", "id": request_id, "payload": {}}, reply_id
    )
    raw = _file_contents(request_id, size)
    await router.route(raw, json.loads(raw))


async def _held(redis: FakeRedis, connection_id: str, reply_id: str) -> list[str]:
    """The envelopes a reply stream holds, read from Redis itself."""
    key = response_stream_key(SESSION, connection_id, reply_id)
    return [fields["envelope"] for _, fields in await redis.xrange(key)]


def _record_refusals(monkeypatch: pytest.MonkeyPatch) -> list[RelayBacklogExceeded]:
    """Every append the relay's Redis script turns away, as the router sees it."""
    refused: list[RelayBacklogExceeded] = []
    real_append = relay_routing.append_response

    async def append(*args: Any, **kwargs: Any) -> None:
        try:
            await real_append(*args, **kwargs)
        except RelayBacklogExceeded as exc:
            refused.append(exc)
            raise

    monkeypatch.setattr(relay_routing, "append_response", append)
    return refused


@pytest.mark.asyncio
async def test_abandoned_reply_streams_stop_at_the_connection_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ten workers each get four answers, enough to fill each of their streams,
    after they stopped reading. The first sixteen answers fill the budget and
    every one after them is turned away. All the relay's keys still expire, so
    a connection that goes quiet leaves nothing behind."""
    monkeypatch.setattr(relay_backlog, "MAX_RESPONSE_BACKLOG_BYTES", STREAM_LIMIT)
    monkeypatch.setattr(
        relay_backlog, "MAX_CONNECTION_RESPONSE_BYTES", CONNECTION_LIMIT
    )
    refused = _record_refusals(monkeypatch)
    async with FakeRedis(decode_responses=True) as redis:
        router = RelayResponseRouter(
            redis, session_id=SESSION, connection_id="connection-1"
        )
        workers = [f"{worker:032x}" for worker in range(10)]
        for worker, reply_id in enumerate(workers):
            for request in range(4):
                await _answer_late(router, reply_id, f"request-{worker}-{request}")

        held = [await _held(redis, "connection-1", reply_id) for reply_id in workers]
        assert [len(envelopes) for envelopes in held] == [4] * 4 + [0] * 6
        assert sum(len(raw.encode()) for raw in sum(held, [])) == CONNECTION_LIMIT
        assert len(refused) == 6 * 4
        for key in await redis.keys("*"):
            assert 0 < await redis.ttl(key) <= STREAM_TTL_SECONDS, key


@pytest.mark.asyncio
async def test_reading_a_reply_stream_gives_its_bytes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker acknowledges each answer it reads, and that answer's bytes
    leave the budget at once, so answers for another worker fit again."""
    monkeypatch.setattr(relay_backlog, "MAX_CONNECTION_RESPONSE_BYTES", 2 * FRAME)
    async with FakeRedis(decode_responses=True) as redis:
        relay = RedisShimRelay(redis)
        presence, _ = await relay._register_presence(
            SESSION, hello={}, user_id="user-1", client_id="autogpt-local-executor"
        )
        connection_id = presence.connection_id
        router = RelayResponseRouter(
            redis, session_id=SESSION, connection_id=connection_id
        )
        reader, waiting = "a" * 32, "b" * 32
        await _answer_late(router, reader, "request-1")
        await _answer_late(router, reader, "request-2")
        await _answer_late(router, waiting, "request-3")
        assert await _held(redis, connection_id, waiting) == []

        transport = RedisRelayTransport(relay, presence, reader)
        async with asyncio.timeout(10), aclosing(transport.iter_text()) as frames:
            read = [json.loads(await anext(frames))["id"] for _ in range(2)]
        assert read == ["request-1", "request-2"]
        assert await _held(redis, connection_id, reader) == []

        for request_id in ("request-4", "request-5", "request-6"):
            await _answer_late(router, waiting, request_id)
        assert [
            json.loads(raw)["id"] for raw in await _held(redis, connection_id, waiting)
        ] == ["request-4", "request-5"]


@pytest.mark.asyncio
async def test_a_reply_stream_that_expires_unread_gives_its_bytes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis tells nobody when a stream expires. The bytes of one that nobody
    read still leave the budget once it has, while a stream that is still
    there keeps its share, so the connection carries on within its budget."""
    monkeypatch.setattr(relay_backlog, "MAX_CONNECTION_RESPONSE_BYTES", 3 * FRAME)
    async with FakeRedis(decode_responses=True) as redis:
        router = RelayResponseRouter(
            redis, session_id=SESSION, connection_id="connection-1"
        )
        kept, abandoned, waiting = "a" * 32, "b" * 32, "c" * 32
        await _answer_late(router, kept, "request-1")
        await _answer_late(router, abandoned, "request-2")
        await _answer_late(router, waiting, "request-3", size=2 * FRAME)
        assert await _held(redis, "connection-1", waiting) == []

        # The abandoned stream's last answer gives it a short life, so it
        # expires unread while the test waits.
        monkeypatch.setattr(relay_backlog, "STREAM_TTL_SECONDS", 1)
        await _answer_late(router, abandoned, "request-4")
        monkeypatch.setattr(relay_backlog, "STREAM_TTL_SECONDS", STREAM_TTL_SECONDS)
        abandoned_key = response_stream_key(SESSION, "connection-1", abandoned)
        async with asyncio.timeout(30):
            while await redis.exists(abandoned_key):
                await asyncio.sleep(0.05)

        await _answer_late(router, waiting, "request-5", size=2 * FRAME)
        await _answer_late(router, waiting, "request-6")
        assert [
            json.loads(raw)["id"] for raw in await _held(redis, "connection-1", waiting)
        ] == ["request-5"]
        assert len(await _held(redis, "connection-1", kept)) == 1
