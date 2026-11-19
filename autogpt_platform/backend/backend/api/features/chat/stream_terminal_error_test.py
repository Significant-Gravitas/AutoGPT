import asyncio
from unittest.mock import AsyncMock

import orjson
import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import RedisError

from backend.api.features.chat.routes import _resume_response
from backend.copilot import stream_registry as registry
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamHeartbeat,
    StreamRecoveredError,
)


async def wire(queue, cursor):
    response = _resume_response("s", queue, cursor=cursor)
    return "".join([part async for part in response.body_iterator])


def parts(content):
    return [
        orjson.loads(line[6:])
        for line in content.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [False, True])
@pytest.mark.parametrize("delivery", ["replay", "live_finish", "live_timeout"])
async def test_lost_error_reaches_response_with_real_turn_and_cursor(
    mocker, cursor, delivery
):
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        real_xadd = redis.xadd

        async def lose_error(key, fields, *args, **kwargs):
            if (
                isinstance(registry._chunk_from_fields(fields), StreamError)
                or delivery == "live_timeout"
            ):
                raise RedisError("lost terminal publication")
            return await real_xadd(key, fields, *args, **kwargs)

        mocker.patch.object(redis, "xadd", side_effect=lose_error)
        await registry.create_session("s", None, "", "", turn_id="t")
        assert await registry.mark_session_completed(
            "s", "provider unavailable", error_code="provider_down", turn_id="t"
        )
        if delivery == "replay":
            queue = await registry.subscribe_to_turn("s", None, "t", "0-0")
        else:
            queue = asyncio.Queue()
            if delivery == "live_timeout":
                mocker.patch.object(redis, "xread", new=AsyncMock(return_value=[]))
            await registry._stream_listener("s", queue, "0-0", turn_id="t")
        content = await wire(queue, cursor)
        decoded = parts(content)
        assert decoded[0] == {
            "type": "data-turn-error",
            "data": {
                "turnId": "t",
                "errorText": "[code:provider_down] provider unavailable",
            },
            "transient": True,
        }
        assert decoded[1] == {
            "type": "error",
            "errorText": "[code:provider_down] provider unavailable",
        }
        assert content.count("data: [DONE]") == 1
        entries = await redis.xrange(registry._get_turn_stream_key("t"))
        ids = [line for line in content.splitlines() if line.startswith("id: ")]
        assert ids == [f"id: t:{entry_id}" for entry_id, _ in entries]
        assert ('"type":"finish"' in content) == (
            delivery != "live_timeout" or not cursor
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_success_and_explicit_stop_never_emit_terminal_error(mocker, stopped):
    mocker.patch.object(registry, "chat_db").return_value.set_turn_duration = (
        AsyncMock()
    )
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="t")
        await registry.mark_session_completed(
            "s",
            registry.CANCELLED_MESSAGE if stopped else None,
            skip_error_publish=stopped,
            turn_id="t",
        )
        queue = await registry.subscribe_to_turn("s", None, "t", "0-0")
        content = await wire(queue, True)
        assert '"type":"finish"' in content
        assert "error" not in content


@pytest.mark.asyncio
async def test_consumed_stored_error_is_not_recovered_in_response(mocker):
    async with FakeRedis(decode_responses=True) as redis:
        mocker.patch.object(registry, "get_redis_async", return_value=redis)
        await registry.create_session("s", None, "", "", turn_id="t")
        await registry.mark_session_completed("s", "failed", turn_id="t")
        entries = await redis.xrange(registry._get_turn_stream_key("t"))
        queue = await registry.subscribe_to_turn("s", None, "t", entries[0][0])
        content = await wire(queue, True)
        assert "error" not in content
        assert f"id: t:{entries[-1][0]}" in content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport", [StreamFinish(), StreamError(errorText="overflow")]
)
async def test_plain_transport_frames_do_not_end_cursor_turn(mocker, transport):
    mocker.patch.object(registry, "unsubscribe_from_session", new=AsyncMock())
    queue = asyncio.Queue()
    queue.put_nowait((None, StreamHeartbeat()))
    queue.put_nowait((None, transport))
    assert await wire(queue, True) == ": heartbeat\n\ndata: [DONE]\n\n"


@pytest.mark.parametrize("message", ["failed", "[code:provider_down] failed"])
def test_recovered_error_retains_strict_legacy_error_shape(message):
    ordinary = StreamError(errorText=message, code="provider_down")
    recovered = StreamRecoveredError(
        turn_id="t", errorText=message, code="provider_down"
    )
    expected = {"type": "error", "errorText": "[code:provider_down] failed"}
    assert parts(ordinary.to_sse()) == [expected]
    assert parts(recovered.to_sse())[1] == expected
    assert recovered.model_dump() == ordinary.model_dump()
    assert isinstance(recovered, StreamError)
    assert "id:" not in recovered.to_sse()
