import asyncio
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import RedisError

from backend.copilot import stream_registry as registry
from backend.copilot.response_model import StreamError, StreamFinish

fakeredis = pytest.importorskip("fakeredis.aioredis")
pytest.importorskip("lupa")


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_lost_publication_recovers_after_next_turn_with_actual_lua(
    mocker, stopped
):
    async with fakeredis.FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        publish = registry.publish_chunk

        async def lose_error(turn_id, chunk, **kwargs):
            if isinstance(chunk, StreamError):
                raise RedisError("lost error publication")
            return await publish(turn_id, chunk, **kwargs)

        mocker.patch.object(registry, "publish_chunk", side_effect=lose_error)
        await registry.create_session("s", None, "", "", turn_id="old")
        assert await registry.mark_session_completed(
            "s",
            registry.CANCELLED_MESSAGE if stopped else "provider unavailable",
            error_code="provider_down",
            skip_error_publish=stopped,
            turn_id="old",
        )
        terminal_key = registry._get_turn_terminal_key("s", "old")
        assert 0 < await redis.ttl(terminal_key) <= registry.config.stream_ttl
        await registry.create_session("s", None, "", "", turn_id="next")
        terminal = await redis.get(terminal_key)
        assert not await registry.mark_session_completed("s", "stale", turn_id="old")
        assert await redis.get(terminal_key) == terminal
        queue = await registry.subscribe_to_turn("s", None, "old", "0-0")
        entries = [queue.get_nowait() for _ in range(queue.qsize())]
        assert isinstance(entries[-1][1], StreamFinish)
        errors = [chunk for _, chunk in entries if isinstance(chunk, StreamError)]
        assert errors == (
            []
            if stopped
            else [StreamError(errorText="provider unavailable", code="provider_down")]
        )
        active = await registry.get_session("s")
        assert active.turn_id == "next" and active.status == "running"


@pytest.mark.asyncio
async def test_resume_consumed_error_with_actual_stream_and_lua(mocker):
    async with fakeredis.FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="t")
        assert await registry.mark_session_completed("s", "failure", turn_id="t")
        entries = await redis.xrange(registry._get_turn_stream_key("t"))
        error_id, _ = entries[0]
        queue = await registry.subscribe_to_turn("s", None, "t", error_id)
        assert queue.qsize() == 1
        assert isinstance(queue.get_nowait()[1], StreamFinish)


@pytest.mark.asyncio
async def test_lost_error_and_finish_recover_at_live_timeout(mocker):
    async with fakeredis.FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="t")
        mocker.patch.object(
            registry, "publish_chunk", new=AsyncMock(side_effect=RedisError("offline"))
        )
        assert await registry.mark_session_completed("s", "failure", turn_id="t")
        # The storage and Lua are real fakeredis operations; only the blocking
        # read timeout is shortened so this test never sleeps for five seconds.
        mocker.patch.object(redis, "xread", new=AsyncMock(return_value=[]))
        queue = asyncio.Queue()
        await registry._stream_listener("s", queue, "0-0", turn_id="t")
        assert queue.get_nowait() == (None, StreamError(errorText="failure"))
        assert queue.get_nowait() == (None, StreamFinish())
        assert queue.empty()
