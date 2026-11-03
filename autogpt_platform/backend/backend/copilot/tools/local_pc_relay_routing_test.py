import json
from typing import Any

import pytest

from backend.copilot.tools import local_pc_relay_routing as relay_routing
from backend.copilot.tools.local_pc_relay_protocol import (
    RelayBacklogExceeded,
    response_stream_key,
)
from backend.copilot.tools.local_pc_relay_routing import RelayResponseRouter
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis


def _message(
    message_type: str,
    message_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"type": message_type, "id": message_id, "payload": payload or {}}


@pytest.mark.asyncio
async def test_correlated_and_status_frames_use_the_worker_reply_stream() -> None:
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

    key = response_stream_key("session-1", "connection-1", reply_id)
    envelopes = [json.loads(fields["envelope"]) for _, fields in redis.streams[key]]
    assert [envelope["type"] for envelope in envelopes] == [
        "STATUS",
        "FILE_CONTENTS",
    ]


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
async def test_status_broadcast_survives_a_reply_stream_nobody_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that finished its request and stopped reading leaves a reply
    stream that fills up. STATUS is advisory, so that full stream must not end
    the shim connection for everyone else, and stops receiving broadcasts."""
    redis = FakeRedis()
    router = RelayResponseRouter(
        redis, session_id="session-1", connection_id="connection-1"
    )
    finished, live = "a" * 32, "b" * 32
    router.register_request(_message("FILE_READ", "request-1"), finished)
    done = _message("FILE_CONTENTS", "request-1", {"content": "ok"})
    await router.route(json.dumps(done), done)
    router.register_request(_message("FILE_READ", "request-2"), live)

    full_key = response_stream_key("session-1", "connection-1", finished)
    attempts: list[str] = []
    real_append = relay_routing.append_response

    async def append(redis_: Any, key: str, raw: str) -> None:
        attempts.append(key)
        if key == full_key:
            raise RelayBacklogExceeded("full")
        await real_append(redis_, key, raw)

    monkeypatch.setattr(relay_routing, "append_response", append)

    status = _message("STATUS", "status-1", {"pending_capacity": 2})
    await router.route(json.dumps(status), status)
    await router.route(json.dumps(status), status)

    live_key = response_stream_key("session-1", "connection-1", live)
    assert [
        json.loads(fields["envelope"])["type"] for _, fields in redis.streams[live_key]
    ] == ["STATUS", "STATUS"]
    assert attempts.count(full_key) == 1


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
