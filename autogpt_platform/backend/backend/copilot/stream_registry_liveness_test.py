"""A running turn's meta is never taken over, and a dead turn never reads running.

Real Redis: the guarantees live in Lua scripts and key TTLs.
"""

import asyncio
import logging
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from backend.copilot import stream_registry
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamHeartbeat,
    StreamStatus,
    StreamTextDelta,
)
from backend.copilot.turn_lease import turn_lease_key
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable


@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop():
    yield


requires_redis = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)


@pytest.fixture
async def session_id():
    session_id = f"liveness-{uuid.uuid4().hex}"
    yield session_id
    redis = await get_redis_async()
    await redis.delete(stream_registry.get_session_meta_key(session_id))


@pytest.fixture
async def turn_ids():
    turn_ids: list[str] = []
    yield turn_ids
    redis = await get_redis_async()
    for turn_id in turn_ids:
        await redis.delete(stream_registry._get_turn_stream_key(turn_id))
        await redis.delete(stream_registry._get_turn_meta_key(turn_id))
        await redis.delete(turn_lease_key(turn_id))


def _turn(turn_ids: list[str]) -> str:
    turn_ids.append(turn_id := str(uuid.uuid4()))
    return turn_id


async def _claim(session_id: str, turn_id: str, *, lease: bool) -> None:
    """What the executor does: take the lease, then publish."""
    redis = await get_redis_async()
    if lease:
        await redis.set(turn_lease_key(turn_id), "pod-a", ex=30)
    await stream_registry.publish_chunk(
        turn_id, StreamStatus(message="Setting up…"), session_id=session_id
    )


@requires_redis
class TestCreateSession:
    async def test_a_running_turn_keeps_its_meta(self, session_id, turn_ids):
        running, late = _turn(turn_ids), _turn(turn_ids)
        await stream_registry.create_session(session_id, "u1", "", "", turn_id=running)

        refused = await stream_registry.create_session(
            session_id, "u1", "", "", turn_id=late
        )

        assert refused.turn_id == running
        meta = await stream_registry.get_session(session_id)
        assert meta is not None and meta.turn_id == running

    async def test_the_same_turn_may_register_again(self, session_id, turn_ids):
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)

        again = await stream_registry.create_session(
            session_id, None, "", "", turn_id=turn_id
        )

        assert again.turn_id == turn_id

    async def test_a_dead_turn_is_ended_and_replaced(self, session_id, turn_ids):
        dead, fresh = _turn(turn_ids), _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=dead)
        await _claim(session_id, dead, lease=False)

        created = await stream_registry.create_session(
            session_id, None, "", "", turn_id=fresh
        )

        assert created.turn_id == fresh
        tail = await _entries(dead)
        assert isinstance(tail[-2], StreamError)
        assert tail[-2].errorText == stream_registry.EXECUTOR_LOST_MESSAGE
        assert isinstance(tail[-1], StreamFinish)

    async def test_a_new_turn_starts_unclaimed(self, session_id, turn_ids):
        first, second = _turn(turn_ids), _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=first)
        await _claim(session_id, first, lease=True)
        redis = await get_redis_async()
        meta_key = stream_registry.get_session_meta_key(session_id)
        await redis.hset(meta_key, "status", "completed")

        await stream_registry.create_session(session_id, None, "", "", turn_id=second)

        assert await redis.hget(meta_key, "claimed") == ""

    async def test_a_refused_dispatch_leaves_the_running_meta(
        self, session_id, turn_ids
    ):
        running, refused = _turn(turn_ids), _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=running)

        await stream_registry.delete_session_meta(session_id, refused)

        meta = await stream_registry.get_session(session_id)
        assert meta is not None and meta.turn_id == running


@requires_redis
class TestExecutorLiveness:
    async def test_a_queued_turn_reads_running_without_a_lease(
        self, session_id, turn_ids
    ):
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
        # The route's own status publish does not claim the turn.
        await stream_registry.publish_chunk(
            turn_id, StreamStatus(message="Message received…")
        )

        active, _ = await stream_registry.get_active_session(session_id)

        assert active is not None and active.turn_id == turn_id

    async def test_a_turn_whose_executor_holds_the_lease_reads_running(
        self, session_id, turn_ids
    ):
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
        await _claim(session_id, turn_id, lease=True)

        active, _ = await stream_registry.get_active_session(session_id)

        assert active is not None and active.turn_id == turn_id

    async def test_a_listener_ends_a_turn_whose_lease_lapsed(
        self, session_id, turn_ids
    ):
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
        await _claim(session_id, turn_id, lease=True)
        queue = await stream_registry.subscribe_to_turn(
            session_id, None, turn_id, "0-0"
        )
        try:
            redis = await get_redis_async()
            await redis.delete(turn_lease_key(turn_id))
            with patch.object(stream_registry, "_LISTENER_POLL_MS", 50):
                served = await _until_finish(queue)
        finally:
            await stream_registry.unsubscribe_from_session(session_id, queue)

        errors = [chunk for _, chunk in served if isinstance(chunk, StreamError)]
        assert [e.errorText for e in errors] == [stream_registry.EXECUTOR_LOST_MESSAGE]
        final_id, final = served[-1]
        assert isinstance(final, StreamFinish) and final_id is not None


@requires_redis
class TestHeartbeats:
    async def test_a_heartbeat_is_not_stored_but_refreshes_the_ttls(
        self, session_id, turn_ids
    ):
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
        redis = await get_redis_async()
        meta_key = stream_registry.get_session_meta_key(session_id)
        await redis.expire(meta_key, 5)

        await stream_registry.publish_chunk(
            turn_id, StreamHeartbeat(), session_id=session_id
        )

        assert await redis.xlen(stream_registry._get_turn_stream_key(turn_id)) == 0
        assert await redis.ttl(meta_key) > 5

    async def test_the_transport_heartbeat_keeps_its_cadence_under_traffic(
        self, session_id, turn_ids, monkeypatch
    ):
        monkeypatch.setattr(stream_registry, "TRANSPORT_HEARTBEAT_INTERVAL_S", 0.2)
        turn_id = _turn(turn_ids)
        await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
        queue = await stream_registry.subscribe_to_turn(
            session_id, None, turn_id, "0-0"
        )
        heartbeats_at: list[float] = []
        try:
            start = time.monotonic()
            while time.monotonic() - start < 1.1:
                await stream_registry.publish_chunk(
                    turn_id, StreamTextDelta(id="t1", delta="x")
                )
                await asyncio.sleep(0.02)
                while not queue.empty():
                    _, chunk = queue.get_nowait()
                    if isinstance(chunk, StreamHeartbeat):
                        heartbeats_at.append(time.monotonic() - start)
        finally:
            await stream_registry.unsubscribe_from_session(session_id, queue)

        assert len(heartbeats_at) >= 4


@pytest.mark.asyncio
async def test_a_disconnect_spares_listeners_started_after_it():
    older = asyncio.create_task(asyncio.sleep(3600))
    newer = asyncio.create_task(asyncio.sleep(3600))
    stream_registry._listener_sessions[901] = ("sess-d", older)
    stream_registry._listener_sessions[902] = ("sess-d", newer)
    stream_registry._listener_started_at[901] = 100.0
    stream_registry._listener_started_at[902] = 300.0
    try:
        cancelled = await stream_registry._cancel_local_listeners("sess-d", 200.0)

        assert cancelled == 1
        assert older.cancelled() and not newer.done()
    finally:
        newer.cancel()
        for qid in (901, 902):
            stream_registry._listener_sessions.pop(qid, None)
            stream_registry._listener_started_at.pop(qid, None)


@pytest.mark.asyncio
async def test_a_disconnect_reaches_the_other_pods():
    with patch.object(
        stream_registry._listener_disconnects, "broadcast", new=AsyncMock()
    ) as broadcast:
        await stream_registry.disconnect_all_listeners("sess-x")

    broadcast.assert_awaited_once_with("sess-x")


def test_an_unknown_chunk_type_is_logged_by_name(caplog):
    with caplog.at_level(logging.WARNING):
        chunk = stream_registry._reconstruct_chunk({"type": "data-from-the-future"})

    assert chunk is None
    assert "data-from-the-future" in caplog.text


async def _entries(turn_id: str) -> list:
    redis = await get_redis_async()
    key = stream_registry._get_turn_stream_key(turn_id)
    raw = await redis.xrange(key)
    [(_, entries)] = stream_registry._stream_entries([(key, raw)])
    return [stream_registry._chunk_from_fields(fields) for _, fields in entries]


async def _until_finish(queue: asyncio.Queue) -> list:
    served = []
    async with asyncio.timeout(10):
        while True:
            entry = await queue.get()
            served.append(entry)
            if isinstance(entry[1], StreamFinish):
                return served
