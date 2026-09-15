import asyncio
from typing import Any

import pytest

from backend.copilot.tools import local_pc_relay_transport as relay_transport
from backend.copilot.tools.local_pc_relay_protocol import (
    EXPECTED_RESPONSE_TYPES,
    PLATFORM_REQUEST_TYPES,
    RelayBacklogExceeded,
    RelayRequestTarget,
    register_pending_request,
    validate_response_correlation,
)
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis
from backend.copilot.tools.local_pc_relay_transport import (
    acknowledge_stream_entry,
    append_request,
    append_response,
)


def _message(message_type: str, message_id: str = "request-1") -> dict[str, Any]:
    return {"type": message_type, "id": message_id, "payload": {}}


def test_every_request_type_has_a_response_contract() -> None:
    assert set(EXPECTED_RESPONSE_TYPES) == set(PLATFORM_REQUEST_TYPES)


def test_response_correlation_preserves_streams_and_unsolicited_events() -> None:
    pending: dict[str, RelayRequestTarget] = {}
    register_pending_request(_message("LOCAL_LLM_COMPLETION"), "a" * 32, pending)

    validate_response_correlation(_message("LOCAL_LLM_COMPLETION_CHUNK"), pending)
    assert pending["request-1"].request_type == "LOCAL_LLM_COMPLETION"
    validate_response_correlation(_message("LOCAL_LLM_COMPLETION_RESPONSE"), pending)
    assert pending == {}

    validate_response_correlation(_message("STATUS", "status-1"), pending)
    validate_response_correlation(
        _message("RECORDING_STEP", "recording-step-1"), pending
    )


def test_response_correlation_rejects_unknown_or_wrong_response() -> None:
    with pytest.raises(ValueError, match="no outstanding request"):
        validate_response_correlation(_message("COMMAND_RESULT"), {})

    pending: dict[str, RelayRequestTarget] = {}
    register_pending_request(_message("FILE_READ"), "a" * 32, pending)
    with pytest.raises(ValueError, match="reused an outstanding correlation id"):
        register_pending_request(_message("FILE_READ"), "a" * 32, pending)
    with pytest.raises(ValueError, match="COMMAND_RESULT for FILE_READ"):
        validate_response_correlation(_message("COMMAND_RESULT"), pending)


@pytest.mark.asyncio
async def test_stream_append_is_atomic_and_consumed_response_releases_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    key = "responses"
    monkeypatch.setattr(relay_transport, "MAX_RESPONSE_BACKLOG_BYTES", 5)

    await append_response(redis, key, "12345")
    message_id, fields = redis.streams[key][0]
    await acknowledge_stream_entry(redis, key, message_id, fields)
    await append_response(redis, key, "abcde")

    assert redis.strings["responses:backlog-bytes"] == 5
    assert len(redis.streams[key]) == 1


@pytest.mark.asyncio
async def test_response_backlog_has_strict_entry_and_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(relay_transport, "MAX_RESPONSE_BACKLOG_ENTRIES", 1)
    await append_response(redis, "entry-limit", "one")
    with pytest.raises(RelayBacklogExceeded):
        await append_response(redis, "entry-limit", "two")

    monkeypatch.setattr(relay_transport, "MAX_RESPONSE_BACKLOG_ENTRIES", 128)
    monkeypatch.setattr(relay_transport, "MAX_RESPONSE_BACKLOG_BYTES", 5)
    await append_response(redis, "byte-limit", "1234")
    with pytest.raises(RelayBacklogExceeded):
        await append_response(redis, "byte-limit", "56")


@pytest.mark.asyncio
async def test_request_append_uses_transactional_ttl_pipeline() -> None:
    redis = FakeRedis()
    await append_request(redis, "requests", "payload", reply_id="a" * 32)

    assert redis.strings["requests:backlog-bytes"] == 7
    assert redis.streams["requests"][0][1] == {
        "envelope": "payload",
        "reply_id": "a" * 32,
        "size": "7",
    }


@pytest.mark.asyncio
async def test_request_backlog_has_a_strict_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(relay_transport, "MAX_REQUEST_BACKLOG_BYTES", 5)

    await append_request(redis, "requests", "1234", reply_id="a" * 32)
    with pytest.raises(RelayBacklogExceeded):
        await append_request(redis, "requests", "56", reply_id="a" * 32)


@pytest.mark.asyncio
async def test_concurrent_appends_cannot_overshoot_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(relay_transport, "MAX_RESPONSE_BACKLOG_BYTES", 5)

    results = await asyncio.gather(
        append_response(redis, "responses", "1234"),
        append_response(redis, "responses", "5678"),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, RelayBacklogExceeded) for result in results) == 1
    assert len(redis.streams["responses"]) == 1
    assert redis.strings["responses:backlog-bytes"] == 4
