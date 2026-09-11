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
    monkeypatch.setattr(relay_routing.time, "monotonic", lambda: now[0])
    router = RelayResponseRouter(
        FakeRedis(), session_id="session-1", connection_id="connection-1"
    )
    router.register_request(_message("FILE_READ", "request-1"), "a" * 32)
    now[0] = relay_routing._REQUEST_CORRELATION_SECONDS + 1
    late = _message("FILE_CONTENTS", "request-1", {"content": "late"})

    with pytest.raises(ValueError, match="no outstanding request"):
        await router.route(json.dumps(late), late)
