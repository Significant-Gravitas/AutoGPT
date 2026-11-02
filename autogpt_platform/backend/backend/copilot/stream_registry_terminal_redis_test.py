import asyncio
from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import RedisError

from backend.copilot import stream_registry as registry
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamRecoveredError,
    StreamTextDelta,
)
from backend.copilot.sdk.session_waiter import wait_for_session_result


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["replay", "missing_finish", "live", "timeout"])
@pytest.mark.parametrize("stopped", [False, True])
async def test_waiter_distinguishes_suppressed_failure_from_success(
    mocker, delivery, stopped
):
    mocker.patch.object(registry, "chat_db").return_value.set_turn_duration = (
        AsyncMock()
    )
    unsubscribe = mocker.spy(registry, "unsubscribe_from_session")
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="t")
        initial = await redis.hgetall(registry.get_session_meta_key("s"))
        await registry.publish_chunk("t", StreamTextDelta(id="text", delta="partial"))
        await registry.mark_session_completed(
            "s",
            registry.CANCELLED_MESSAGE if stopped else None,
            skip_error_publish=stopped,
            turn_id="t",
        )
        entries = await redis.xrange(registry._get_turn_stream_key("t"))
        if delivery == "missing_finish":
            await redis.xdel(registry._get_turn_stream_key("t"), entries[-1][0])
        elif delivery in {"live", "timeout"}:
            mocker.patch.object(registry, "_read_meta_with_retry", return_value=initial)
            mocker.patch.object(
                redis,
                "xread",
                new=AsyncMock(
                    side_effect=[
                        [("stream", entries[:-1])],
                        [("stream", entries[-1:])] if delivery == "live" else [],
                    ]
                ),
            )
        outcome, result = await wait_for_session_result(
            session_id="s",
            user_id=None,
            timeout=5,
        )

        assert outcome == ("failed" if stopped else "completed")
        assert result.response_text == "partial"
        unsubscribe.assert_awaited_once()
        assert registry._listener_sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_lost_publication_recovers_after_next_turn_with_actual_lua(
    mocker, stopped
):
    async with FakeRedis(decode_responses=True) as redis:
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
            else [
                StreamRecoveredError(
                    turn_id="old",
                    errorText="provider unavailable",
                    code="provider_down",
                )
            ]
        )
        active = await registry.get_session("s")
        assert active.turn_id == "next" and active.status == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("successor_status", ["running", "completed", "failed"])
async def test_internal_failure_remains_bound_to_subscribed_turn(
    mocker, successor_status
):
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="old")
        pinned = await redis.hgetall(registry.get_session_meta_key("s"))
        await registry.mark_session_completed(
            "s", registry.CANCELLED_MESSAGE, skip_error_publish=True, turn_id="old"
        )
        await registry.create_session("s", None, "", "", turn_id="next")
        await redis.hset(
            registry.get_session_meta_key("s"),
            mapping={
                "status": successor_status,
                "error_message": "successor failure",
            },
        )
        mocker.patch.object(registry, "_read_meta_with_retry", return_value=pinned)

        outcome, _ = await wait_for_session_result(
            session_id="s", user_id=None, timeout=5
        )

        assert outcome == "failed"
        public = await registry.subscribe_to_turn("s", None, "old", "0-0")
        assert [type(public.get_nowait()[1]) for _ in range(public.qsize())] == [
            StreamFinish
        ]


@pytest.mark.asyncio
async def test_internal_waiter_does_not_infer_failure_from_successor(mocker):
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="next")
        await redis.hset(
            registry.get_session_meta_key("s"),
            mapping={
                "status": "failed",
                "error_message": "another turn's failure",
            },
        )
        assert (
            await registry._recover_terminal_error(
                redis,
                "s",
                "old",
                "0-0",
                include_suppressed_errors=True,
            )
            is None
        )


@pytest.mark.asyncio
async def test_resume_consumed_error_with_actual_stream_and_lua(mocker):
    async with FakeRedis(decode_responses=True) as redis:
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
    async with FakeRedis(decode_responses=True) as redis:
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
        assert queue.get_nowait() == (
            None,
            StreamRecoveredError(turn_id="t", errorText="failure"),
        )
        assert queue.get_nowait() == (None, StreamFinish())
        assert queue.empty()
