import asyncio
from unittest.mock import AsyncMock

import orjson
import pytest
from redis.crc import key_slot
from redis.exceptions import RedisError

from backend.copilot import stream_registry as registry
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamRecoveredError,
)
from backend.copilot.stream_terminal import TurnTerminal


@pytest.fixture
def redis(mocker):
    client = AsyncMock()
    client.hgetall.return_value = {
        "session_id": "s",
        "user_id": "u",
        "turn_id": "t",
        "status": "failed",
        "error_message": "provider unavailable",
        "error_code": "provider_down",
    }
    client.get.return_value = None
    client.hmget.return_value = ["failed", "t"]
    client.xread.return_value = []
    client.xrevrange.return_value = []
    client.xinfo_stream.return_value = {"entries-added": 1, "length": 1}
    mocker.patch.object(registry, "get_redis_async", return_value=client)
    return client


def stored(chunk, entry_id="2-0"):
    return [("stream", [(entry_id, {"data": chunk.model_dump_json()})])]


def drain(queue):
    return [queue.get_nowait() for _ in range(queue.qsize())]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["session", "turn", "live"])
@pytest.mark.parametrize("finish_stored", [False, True])
async def test_failed_turn_recovers_error_before_finish(redis, path, finish_stored):
    redis.xread.side_effect = [stored(StreamFinish()) if finish_stored else [], []]
    if path == "session":
        queue = await registry.subscribe_to_session("s", "u")
    elif path == "turn":
        queue = await registry.subscribe_to_turn("s", "u", "t", "0-0")
        if not finish_stored:
            await asyncio.wait_for(registry._listener_sessions[id(queue)][1], 2)
    else:
        queue = asyncio.Queue()
        await registry._stream_listener("s", queue, "0-0", turn_id="t")
    entries = drain(queue)
    assert [type(chunk) for _, chunk in entries] == [StreamRecoveredError, StreamFinish]
    assert entries[0][1].turn_id == "t"
    assert entries[0][1].code == "provider_down"
    assert entries[0][1].errorText == "provider unavailable"
    assert entries[0][0] is None


@pytest.mark.asyncio
async def test_completion_persists_error_with_ownership_transition(redis, mocker):
    redis.hgetall.return_value.update(status="running", user_id="")
    cas = mocker.patch.object(registry, "hash_compare_and_set", return_value=True)
    publish = mocker.patch.object(registry, "publish_chunk", new=AsyncMock())
    assert await registry.mark_session_completed(
        "s", "provider unavailable", error_code="provider_down", turn_id="t"
    )
    args = cas.call_args.kwargs
    assert args["guard"] == ("turn_id", "t")
    key, payload, ttl = args["record"]
    assert key == registry._get_turn_terminal_key("s", "t")
    assert orjson.loads(payload)["error_message"] == "provider unavailable"
    assert ttl == registry.config.stream_ttl
    assert [type(call.args[1]) for call in publish.await_args_list] == [
        StreamError,
        StreamFinish,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
async def test_terminal_lookup_failure_keeps_the_stored_finish(redis, live):
    redis.get.side_effect = RedisError("terminal lookup interrupted")
    redis.hgetall.return_value.update(status="completed", error_message="")
    redis.xread.side_effect = [stored(StreamFinish()), []]
    if live:
        queue = asyncio.Queue()
        await registry._stream_listener("s", queue, "0-0", turn_id="t")
    else:
        queue = await registry.subscribe_to_session("s", "u")
    assert drain(queue) == [("t:2-0", StreamFinish())]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["session", "turn", "live"])
@pytest.mark.parametrize("finish_stored", [False, True])
async def test_resume_after_consumed_error_never_duplicates(redis, path, finish_stored):
    error = StreamError(errorText="provider unavailable", code="provider_down")
    redis.xrevrange.return_value = [
        (b"1-0", {b"data": error.model_dump_json().encode()})
    ]
    redis.xread.side_effect = [stored(StreamFinish()) if finish_stored else [], []]
    if path == "session":
        queue = await registry.subscribe_to_session("s", "u", "1-0")
    elif path == "turn":
        queue = await registry.subscribe_to_turn("s", "u", "t", "1-0")
        if not finish_stored:
            await asyncio.wait_for(registry._listener_sessions[id(queue)][1], 2)
    else:
        queue = asyncio.Queue()
        await registry._stream_listener("s", queue, "1-0", turn_id="t")
    assert [type(chunk) for _, chunk in drain(queue)] == [StreamFinish]
    redis.xrevrange.assert_awaited_once()
    assert redis.xrevrange.call_args.kwargs["max"] == "1-0"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["session", "turn", "live"])
async def test_replayed_error_is_not_synthesized_again(redis, path):
    error = StreamError(errorText="provider unavailable", code="provider_down")
    redis.xread.side_effect = [stored(error, "1-0") + stored(StreamFinish()), []]
    if path == "session":
        queue = await registry.subscribe_to_session("s", "u")
    elif path == "turn":
        queue = await registry.subscribe_to_turn("s", "u", "t", "0-0")
    else:
        queue = asyncio.Queue()
        await registry._stream_listener("s", queue, "0-0", turn_id="t")
    assert drain(queue) == [("t:1-0", error), ("t:2-0", StreamFinish())]
    redis.get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("encode", [False, True])
async def test_per_turn_record_survives_successor_and_decodes(redis, encode):
    terminal = orjson.dumps(redis.hgetall.return_value)
    redis.get.return_value = terminal if encode else terminal.decode()
    redis.hgetall.return_value = {"turn_id": "next", "status": "running"}
    error = await registry._recover_terminal_error(redis, "s", "t", "0-0")
    assert error == StreamRecoveredError(
        turn_id="t", errorText="provider unavailable", code="provider_down"
    )
    redis.get.return_value = None
    assert await registry._recover_terminal_error(redis, "s", "t", "0-0") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        None,
        b"bad json",
        b"[]",
        b'{"turn_id":"other"}',
        b'{"turn_id":"t","status":"failed","error_message":42}',
        b'{"turn_id":"t","status":"failed","error_code":false}',
        b'{"turn_id":"t","status":"running","error_message":"not terminal"}',
    ],
)
async def test_missing_or_invalid_record_falls_back_to_matching_hash(redis, raw):
    redis.get.return_value = raw
    redis.hgetall.return_value = {
        key.encode(): value.encode()
        for key, value in redis.hgetall.return_value.items()
    }
    assert (
        await registry._recover_terminal_error(redis, "s", "t", "0-0")
    ).code == "provider_down"


@pytest.mark.asyncio
async def test_error_cursor_read_failure_preserves_finish(redis):
    redis.xrevrange.side_effect = RedisError("cursor unavailable")
    redis.xread.side_effect = [stored(StreamFinish()), []]
    queue = await registry.subscribe_to_turn("s", "u", "t", "1-0")
    assert drain(queue) == [("t:2-0", StreamFinish())]


@pytest.mark.asyncio
@pytest.mark.parametrize("entries", [None, [(b"1-0", None)]])
async def test_missing_reverse_scan_entries_keep_terminal_recovery(redis, entries):
    redis.xrevrange.return_value = entries
    assert await registry._recover_terminal_error(
        redis, "s", "t", "1-0"
    ) == StreamRecoveredError(
        turn_id="t", errorText="provider unavailable", code="provider_down"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_error", [False, True])
async def test_consumed_error_scan_continues_with_exclusive_cursor(
    redis, mocker, stored_error
):
    mocker.patch.object(registry.config, "stream_replay_count", 1)
    finish = {"data": StreamFinish().model_dump_json()}
    error = {"data": StreamError(errorText="provider unavailable").model_dump_json()}
    redis.xrevrange.side_effect = [
        [("3-0", finish)],
        [("2-0", error if stored_error else finish)],
        [],
    ]
    result = await registry._recover_terminal_error(redis, "s", "t", "3-0")
    assert (result is None) == stored_error
    assert [call.kwargs["max"] for call in redis.xrevrange.await_args_list] == (
        ["3-0", "(3-0"] if stored_error else ["3-0", "(3-0", "(2-0"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_publish", ["error", "finish", "both"])
async def test_publication_failure_leaves_recoverable_terminal(
    redis, mocker, failed_publish
):
    redis.hgetall.return_value.update(status="running", user_id="")
    entries = []

    async def commit(*args, **kwargs):
        redis.get.return_value = kwargs["record"][1]
        redis.hgetall.return_value.update(status="failed", **kwargs["updates"])
        return True

    async def xadd(key, fields):
        chunk = registry._chunk_from_fields(fields)
        if (
            failed_publish == "both"
            or (failed_publish == "error" and isinstance(chunk, StreamError))
            or (failed_publish == "finish" and isinstance(chunk, StreamFinish))
        ):
            raise RedisError("publication interrupted")
        entry_id = f"{len(entries) + 1}-0"
        entries.append((entry_id, fields))
        return entry_id

    mocker.patch.object(registry, "hash_compare_and_set", side_effect=commit)
    redis.xadd.side_effect = xadd
    assert await registry.mark_session_completed(
        "s", "provider unavailable", turn_id="t"
    )
    redis.xread.side_effect = [[("stream", entries)], []]
    queue = await registry.subscribe_to_session("s", None)
    assert [type(chunk) for _, chunk in drain(queue)] == [
        StreamError if failed_publish == "finish" else StreamRecoveredError,
        StreamFinish,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_rejected_or_failed_persistence_cannot_publish(redis, mocker, failure):
    cas = mocker.patch.object(registry, "hash_compare_and_set", return_value=False)
    if failure:
        cas.side_effect = RedisError("persistence interrupted")
    publish = mocker.patch.object(registry, "publish_chunk")
    if failure:
        with pytest.raises(RedisError):
            await registry.mark_session_completed("s", "failed", turn_id="t")
    else:
        assert not await registry.mark_session_completed("s", "failed", turn_id="t")
    publish.assert_not_called()
    redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_stop_record_cannot_resurrect_cancellation(redis, mocker):
    redis.hgetall.return_value.update(status="running", user_id="")
    cas = mocker.patch.object(registry, "hash_compare_and_set", return_value=True)
    publish = mocker.patch.object(registry, "publish_chunk")
    await registry.mark_session_completed(
        "s", registry.CANCELLED_MESSAGE, skip_error_publish=True, turn_id="t"
    )
    terminal = orjson.loads(cas.call_args.kwargs["record"][1])
    assert TurnTerminal.model_validate(terminal).as_error() is None
    assert [type(call.args[1]) for call in publish.await_args_list] == [StreamFinish]


def test_terminal_keys_are_per_turn_and_share_session_cluster_slot():
    first = registry._get_turn_terminal_key("s", "one")
    second = registry._get_turn_terminal_key("s", "two")
    assert first != second
    assert (
        key_slot(first.encode())
        == key_slot(second.encode())
        == key_slot(registry.get_session_meta_key("s").encode())
    )


@pytest.mark.asyncio
async def test_new_turn_clears_session_error_fields(redis):
    await registry.create_session("s", "u", "", "", turn_id="next")
    mapping = redis.hset.call_args_list[0].kwargs["mapping"]
    assert mapping["error_message"] == mapping["error_code"] == ""
    assert mapping["turn_id"] == "next"


@pytest.mark.asyncio
async def test_replayed_error_stays_delivered_into_live_timeout(redis):
    redis.hgetall.return_value["status"] = "running"
    error = StreamError(errorText="provider unavailable", code="provider_down")
    redis.xread.side_effect = [stored(error, "1-0"), []]
    queue = await registry.subscribe_to_turn("s", "u", "t", "0-0")
    await asyncio.wait_for(registry._listener_sessions[id(queue)][1], 2)
    assert drain(queue) == [("t:1-0", error), (None, StreamFinish())]
    redis.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_turn_owned_by_someone_else_is_not_recovered(redis):
    assert await registry.subscribe_to_session("s", "other") is None
    with pytest.raises(registry.TurnStreamGone):
        await registry.subscribe_to_turn("s", "other", "t", "0-0")
    redis.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_trimmed_resume_still_requires_checkpoint(redis):
    redis.xinfo_stream.return_value = {
        "entries-added": 4,
        "length": 2,
        "first-entry": ("3-0", {"data": StreamFinish().model_dump_json()}),
    }
    with pytest.raises(registry.TurnStreamTrimmed):
        await registry.subscribe_to_turn("s", "u", "t", "1-0")
    redis.get.assert_not_awaited()
