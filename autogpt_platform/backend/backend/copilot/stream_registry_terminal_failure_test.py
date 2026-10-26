import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from redis.exceptions import RedisError

from backend.copilot import stream_registry
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
async def test_terminal_lookup_failure_keeps_the_stored_finish(live):
    redis = AsyncMock()
    redis.hgetall.return_value = {
        "session_id": "s",
        "user_id": "u",
        "turn_id": "t",
        "status": "completed",
    }
    redis.get.side_effect = RedisError("terminal lookup interrupted")
    finish = stream_registry.StreamFinish()
    redis.xread.side_effect = [
        [("stream", [("1-0", {"data": finish.model_dump_json()})])],
        [],
    ]
    redis.hmget.return_value = ["completed", "t"]
    with patch.object(
        stream_registry, "get_redis_async", new=AsyncMock(return_value=redis)
    ):
        if live:
            queue = asyncio.Queue()
            await stream_registry._stream_listener("s", queue, "0-0", turn_id="t")
        else:
            queue = await stream_registry.subscribe_to_session("s", "u")
    assert queue is not None
    assert queue.get_nowait() == ("t:1-0", finish)
    assert queue.empty()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no isolated Redis configured",
)
@pytest.mark.parametrize("stopped", [False, True])
async def test_terminal_recovery_and_user_stop_on_real_redis(stopped):
    session_id = f"desktop-terminal-{uuid.uuid4().hex}"
    old_turn, new_turn = str(uuid.uuid4()), str(uuid.uuid4())
    redis = await redis_client.get_redis_async()
    publish = stream_registry.publish_chunk

    async def lose_error(turn_id, chunk, **kwargs):
        if isinstance(chunk, stream_registry.StreamError):
            raise RedisError("lost error publication")
        return await publish(turn_id, chunk, **kwargs)

    try:
        await stream_registry.create_session(session_id, None, "", "", turn_id=old_turn)
        with (
            patch.object(stream_registry, "publish_chunk", new=lose_error),
            patch.object(
                stream_registry.chat_db(),
                "set_turn_duration",
                new=AsyncMock(),
                create=True,
            ),
        ):
            assert await stream_registry.mark_session_completed(
                session_id,
                error_message=(
                    "Operation cancelled" if stopped else "provider unavailable"
                ),
                error_code="provider_unavailable",
                skip_error_publish=stopped,
                turn_id=old_turn,
            )
        terminal_key = stream_registry._get_turn_terminal_key(session_id, old_turn)
        assert 0 < await redis.ttl(terminal_key) <= stream_registry.config.stream_ttl
        await stream_registry.create_session(session_id, None, "", "", turn_id=new_turn)
        queue = asyncio.Queue()
        await asyncio.wait_for(
            stream_registry._stream_listener(
                session_id, queue, "0-0", turn_id=old_turn
            ),
            timeout=10,
        )
        entries = [queue.get_nowait() for _ in range(queue.qsize())]
        assert isinstance(entries[-1][1], stream_registry.StreamFinish)
        assert entries[-1][0] is not None
        errors = [
            chunk
            for _, chunk in entries
            if isinstance(chunk, stream_registry.StreamError)
        ]
        if stopped:
            assert not errors
        else:
            assert len(errors) == 1
            assert errors[0].code == "provider_unavailable"
            assert errors[0].errorText == "provider unavailable"
        active = await stream_registry.get_session(session_id)
        assert (
            active is not None
            and active.turn_id == new_turn
            and active.status == "running"
        )
    finally:
        await redis.delete(stream_registry.get_session_meta_key(session_id))
        for turn_id in (old_turn, new_turn):
            for key in (
                stream_registry._get_turn_stream_key(turn_id),
                stream_registry._get_turn_meta_key(turn_id),
                stream_registry._get_turn_terminal_key(session_id, turn_id),
            ):
                await redis.delete(key)
