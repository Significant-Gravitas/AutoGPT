import json
from typing import Any

import pytest

from backend.copilot.tools import local_pc_relay_routing as relay_routing
from backend.copilot.tools.local_pc_relay_backlog import STREAM_TTL_SECONDS
from backend.copilot.tools.local_pc_relay_protocol import (
    MAX_STATUS_BYTES,
    RelayBacklogExceeded,
    response_stream_key,
    status_stream_key,
)
from backend.copilot.tools.local_pc_relay_routing import RelayResponseRouter
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis


def _message(
    message_type: str,
    message_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"type": message_type, "id": message_id, "payload": payload or {}}


def _status_envelopes(redis: FakeRedis) -> list[dict[str, Any]]:
    key = status_stream_key("session-1", "connection-1")
    return [json.loads(fields["envelope"]) for _, fields in redis.streams.get(key, [])]


def _status_bytes_held(redis: FakeRedis) -> int:
    """The bytes of every STATUS frame in any of the fake's streams."""
    return sum(
        len(fields["envelope"].encode())
        for entries in redis.streams.values()
        for _, fields in entries
        if json.loads(fields["envelope"])["type"] == "STATUS"
    )


@pytest.mark.asyncio
async def test_a_response_uses_its_reply_stream_and_status_the_connection() -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    reply_id = "a" * 32
    router.register_request(_message("FILE_READ", "request-1"), reply_id)

    status = _message("STATUS", "status-1", {"pending_capacity": 2})
    await router.route(json.dumps(status), status)
    response = _message("FILE_CONTENTS", "request-1", {"content": "ok"})
    await router.route(json.dumps(response), response)

    assert _stream_types(redis, reply_id) == ["FILE_CONTENTS"]
    assert [envelope["id"] for envelope in _status_envelopes(redis)] == ["status-1"]
    assert redis.expirations[status_stream_key("session-1", "connection-1")] == (
        STREAM_TTL_SECONDS
    )


@pytest.mark.asyncio
async def test_recording_steps_follow_start_owner_and_unknown_steps_are_dropped() -> (
    None
):
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    unknown = _message("RECORDING_STEP", "step-0", {"recording_id": "unknown"})
    await router.route(json.dumps(unknown), unknown)
    assert redis.streams == {}

    reply_id = "b" * 32
    router.register_request(_message("START_RECORDING", "start-1"), reply_id)
    started = _message("RECORDING_STARTED", "start-1", {"recording_id": "recording-1"})
    await router.route(json.dumps(started), started)
    step = _message("RECORDING_STEP", "step-1", {"recording_id": "recording-1"})
    await router.route(json.dumps(step), step)

    key = response_stream_key("session-1", "connection-1", reply_id)
    assert [
        json.loads(fields["envelope"])["type"] for _, fields in redis.streams[key]
    ] == ["RECORDING_STARTED", "RECORDING_STEP"]


def test_pending_requests_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    router = RelayResponseRouter(
        FakeRedis(), session_id="session-1", connection_id="connection-1"
    )
    monkeypatch.setattr(relay_routing, "_MAX_PENDING_REQUESTS", 1)
    router.register_request(_message("FILE_READ", "request-1"), "a" * 32)

    with pytest.raises(RelayBacklogExceeded):
        router.register_request(_message("FILE_READ", "request-2"), "b" * 32)


@pytest.mark.asyncio
async def test_expired_request_correlation_is_pruned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(relay_routing, "monotonic", lambda: now[0])
    router = RelayResponseRouter(
        FakeRedis(), session_id="session-1", connection_id="connection-1"
    )
    router.register_request(_message("FILE_READ", "request-1"), "a" * 32)
    now[0] = relay_routing._REQUEST_CORRELATION_SECONDS + 1
    late = _message("FILE_CONTENTS", "request-1", {"content": "late"})

    with pytest.raises(ValueError, match="no outstanding request"):
        await router.route(json.dumps(late), late)


@pytest.mark.asyncio
async def test_status_is_held_once_however_many_requests_the_connection_answered() -> (
    None
):
    """A machine request opens a reply stream, reads its answer and stops
    reading. STATUS needs no request, so copying it into every recent reply
    stream held each frame once per answered request, in streams nobody
    reads. The connection holds one STATUS, the latest."""
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    answered = [f"{index:032x}" for index in range(25)]
    for index, reply_id in enumerate(answered):
        router.register_request(
            _message("DIRECTORY_LIST_REQUEST", f"request-{index}"), reply_id
        )
        done = _message("DIRECTORY_LIST_RESPONSE", f"request-{index}", {"entries": []})
        await router.route(json.dumps(done), done)
    outstanding = "f" * 32
    router.register_request(_message("FILE_READ", "request-open"), outstanding)

    frames = [
        json.dumps(
            _message("STATUS", f"status-{capacity}", {"pending_capacity": capacity})
            | {"pad": "x" * 4096}
        )
        for capacity in (0, 1, 3)
    ]
    for raw in frames:
        await router.route(raw, json.loads(raw))

    assert _status_bytes_held(redis) == len(frames[-1].encode())
    [kept] = _status_envelopes(redis)
    assert kept["payload"] == {"pending_capacity": 3}
    for reply_id in answered:
        assert _stream_types(redis, reply_id) == ["DIRECTORY_LIST_RESPONSE"]
    assert _stream_types(redis, outstanding) == []


def _status_of_size(size: int, character: str = "x") -> str:
    """A STATUS frame of exactly ``size`` UTF-8 bytes."""
    empty = json.dumps(_message("STATUS", "status-big", {"pad": ""}))
    width = len(character.encode())
    assert (size - len(empty)) % width == 0
    padded = _message(
        "STATUS", "status-big", {"pad": character * ((size - len(empty)) // width)}
    )
    raw = json.dumps(padded, ensure_ascii=False)
    assert len(raw.encode()) == size
    return raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        _status_of_size(MAX_STATUS_BYTES + 1),
        # Fewer characters than the limit, more bytes.
        _status_of_size(MAX_STATUS_BYTES + 2, "é"),
    ],
    ids=["one-byte-characters", "two-byte-characters"],
)
async def test_a_status_frame_over_the_limit_is_dropped_and_the_connection_carries_on(
    raw: str,
) -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    reply_id = "a" * 32
    router.register_request(_message("FILE_READ", "request-1"), reply_id)
    earlier = _message("STATUS", "status-1", {"pending_capacity": 1})
    await router.route(json.dumps(earlier), earlier)

    await router.route(raw, json.loads(raw))
    response = _message("FILE_CONTENTS", "request-1", {"content": "ok"})
    await router.route(json.dumps(response), response)

    assert [envelope["id"] for envelope in _status_envelopes(redis)] == ["status-1"]
    assert _stream_types(redis, reply_id) == ["FILE_CONTENTS"]


@pytest.mark.asyncio
async def test_a_status_frame_at_the_limit_is_kept() -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    raw = _status_of_size(MAX_STATUS_BYTES)

    await router.route(raw, json.loads(raw))

    assert [envelope["id"] for envelope in _status_envelopes(redis)] == ["status-big"]


def _full_stream(monkeypatch: pytest.MonkeyPatch, full_keys: set[str]) -> list[str]:
    """Make append_response refuse the streams in ``full_keys``, as Redis does
    once a reply stream hits its backlog cap. Returns every attempted key."""
    attempts: list[str] = []
    real_append = relay_routing.append_response

    async def append(redis_: Any, key: str, raw: str) -> None:
        attempts.append(key)
        if key in full_keys:
            raise RelayBacklogExceeded("full")
        await real_append(redis_, key, raw)

    monkeypatch.setattr(relay_routing, "append_response", append)
    return attempts


def _stream_types(redis: FakeRedis, reply_id: str) -> list[str]:
    key = response_stream_key("session-1", "connection-1", reply_id)
    return [
        json.loads(fields["envelope"])["type"]
        for _, fields in redis.streams.get(key, [])
    ]


@pytest.mark.asyncio
async def test_a_full_reply_stream_drops_its_response_and_keeps_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim's answer to a worker that stopped reading must not end the
    shim connection, which would fail every other worker's requests."""
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    stalled, live = "a" * 32, "b" * 32
    router.register_request(_message("FILE_READ", "request-1"), stalled)
    router.register_request(_message("FILE_READ", "request-2"), live)
    _full_stream(
        monkeypatch, {response_stream_key("session-1", "connection-1", stalled)}
    )

    dropped = _message("FILE_CONTENTS", "request-1", {"content": "lost"})
    await router.route(json.dumps(dropped), dropped)
    answered = _message("FILE_CONTENTS", "request-2", {"content": "ok"})
    await router.route(json.dumps(answered), answered)

    assert _stream_types(redis, stalled) == []
    assert _stream_types(redis, live) == ["FILE_CONTENTS"]


@pytest.mark.asyncio
async def test_a_stream_that_loses_a_chunk_drops_the_rest_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a chunk is dropped, later chunks and the final response are dropped
    too, even if the worker drains its stream: it must time out rather than
    read a completion with a hole in it. Its next request is unaffected."""
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    reply_id = "a" * 32
    key = response_stream_key("session-1", "connection-1", reply_id)
    router.register_request(_message("LOCAL_LLM_COMPLETION", "request-1"), reply_id)
    full = {key}
    _full_stream(monkeypatch, full)

    first = _message("LOCAL_LLM_COMPLETION_CHUNK", "request-1", {"delta": "a"})
    await router.route(json.dumps(first), first)
    full.clear()
    second = _message("LOCAL_LLM_COMPLETION_CHUNK", "request-1", {"delta": "b"})
    await router.route(json.dumps(second), second)
    final = _message("LOCAL_LLM_COMPLETION_RESPONSE", "request-1", {"text": "ab"})
    await router.route(json.dumps(final), final)
    assert _stream_types(redis, reply_id) == []

    router.register_request(_message("FILE_READ", "request-2"), reply_id)
    answered = _message("FILE_CONTENTS", "request-2", {"content": "ok"})
    await router.route(json.dumps(answered), answered)
    assert _stream_types(redis, reply_id) == ["FILE_CONTENTS"]


@pytest.mark.asyncio
async def test_a_recording_step_for_a_full_stream_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    reply_id = "b" * 32
    router.register_request(_message("START_RECORDING", "start-1"), reply_id)
    started = _message("RECORDING_STARTED", "start-1", {"recording_id": "rec-1"})
    await router.route(json.dumps(started), started)
    _full_stream(
        monkeypatch, {response_stream_key("session-1", "connection-1", reply_id)}
    )

    step = _message("RECORDING_STEP", "step-1", {"recording_id": "rec-1"})
    await router.route(json.dumps(step), step)

    assert _stream_types(redis, reply_id) == ["RECORDING_STARTED"]


def _request(message_id: str) -> str:
    return json.dumps(_message("FILE_READ", message_id, {"path": "a.txt"}))


def _stream_envelopes(redis: FakeRedis, reply_id: str) -> list[dict[str, Any]]:
    key = response_stream_key("session-1", "connection-1", reply_id)
    return [json.loads(fields["envelope"]) for _, fields in redis.streams.get(key, [])]


@pytest.mark.asyncio
async def test_a_request_over_the_cap_gets_an_error_and_the_others_carry_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(relay_routing, "_MAX_PENDING_REQUESTS", 1)
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    first, second = "a" * 32, "b" * 32

    assert await router.accept_request(_request("request-1"), first) is not None
    assert await router.accept_request(_request("request-2"), second) is None

    [refusal] = _stream_envelopes(redis, second)
    assert refusal["type"] == "ERROR"
    assert refusal["id"] == "request-2"
    assert refusal["payload"]["code"] == "SHIM_OVERLOADED"
    # A capacity hint would hold back the worker's later requests.
    assert "pending_capacity" not in refusal["payload"]
    answered = _message("FILE_CONTENTS", "request-1", {"content": "ok"})
    await router.route(json.dumps(answered), answered)
    assert _stream_types(redis, first) == ["FILE_CONTENTS"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope", "reply_id"),
    [
        ("not json", "a" * 32),
        (json.dumps(_message("COMMAND_RESULT", "request-1")), "a" * 32),
        (_request("request-1"), None),
        (_request("request-1"), "not-a-reply-id"),
    ],
)
async def test_a_malformed_request_is_dropped_without_ending_anything(
    envelope: str, reply_id: str | None
) -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )

    assert await router.accept_request(envelope, reply_id) is None
    assert redis.streams == {}


@pytest.mark.asyncio
async def test_a_reused_id_keeps_the_outstanding_request() -> None:
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    original, duplicate = "a" * 32, "b" * 32
    assert await router.accept_request(_request("request-1"), original) is not None

    assert await router.accept_request(_request("request-1"), duplicate) is None

    answered = _message("FILE_CONTENTS", "request-1", {"content": "ok"})
    await router.route(json.dumps(answered), answered)
    assert _stream_types(redis, original) == ["FILE_CONTENTS"]
    assert _stream_types(redis, duplicate) == []


@pytest.mark.asyncio
async def test_a_dropped_recording_start_records_no_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that never saw RECORDING_STARTED does not own the recording,
    so its steps are not appended to the stream that just refused it."""
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    reply_id = "b" * 32
    full_key = response_stream_key("session-1", "connection-1", reply_id)
    router.register_request(_message("START_RECORDING", "start-1"), reply_id)
    attempts = _full_stream(monkeypatch, {full_key})

    started = _message("RECORDING_STARTED", "start-1", {"recording_id": "rec-1"})
    await router.route(json.dumps(started), started)
    step = _message("RECORDING_STEP", "step-1", {"recording_id": "rec-1"})
    await router.route(json.dumps(step), step)

    assert attempts == [full_key]
