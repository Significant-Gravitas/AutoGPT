"""Shared recording lifecycle state for Local PC executor processes."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.data.redis_client import get_redis_async

_STATE_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_RECORDINGS_PER_SESSION = 100

_STORE_STATE_SCRIPT = """
-- LOCAL_EXECUTOR_STORE_RECORDING_STATE
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[1])

local overflow = redis.call('ZCARD', KEYS[2]) - tonumber(ARGV[4])
local stale_ids = {}
if overflow > 0 then
    stale_ids = redis.call('ZRANGE', KEYS[2], 0, overflow - 1)
    for _, recording_id in ipairs(stale_ids) do
        redis.call('ZREM', KEYS[2], recording_id)
        redis.call('HDEL', KEYS[1], recording_id)
    end
end

redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
return #stale_ids
"""


class RecordingState(BaseModel):
    recording_id: str
    status: Literal["recording", "stopped", "reviewed"]
    mode: str | None = None
    interpretation_route: str | None = None
    channels: list[str] = Field(default_factory=list)
    summary: dict[str, Any] | None = None
    step_count: int | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    reviewed_at: float | None = None


def _session_tag(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:32]


def _recordings_key(session_id: str) -> str:
    tag = _session_tag(session_id)
    return f"local-executor:{{{tag}}}:recording-state"


def _recording_order_key(session_id: str) -> str:
    tag = _session_tag(session_id)
    return f"local-executor:{{{tag}}}:recording-order"


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


async def get_recording_state(
    session_id: str, recording_id: str
) -> RecordingState | None:
    redis: Any = await get_redis_async()
    raw = await redis.hget(_recordings_key(session_id), recording_id)
    if raw is None:
        return None
    try:
        return RecordingState.model_validate_json(_decode(raw))
    except ValueError:
        return None


async def list_recording_states(session_id: str) -> list[RecordingState]:
    redis: Any = await get_redis_async()
    ids = await redis.zrevrange(
        _recording_order_key(session_id), 0, _MAX_RECORDINGS_PER_SESSION - 1
    )
    if not ids:
        return []
    decoded_ids = [_decode(recording_id) for recording_id in ids]
    values = await redis.hmget(_recordings_key(session_id), decoded_ids)
    states: list[RecordingState] = []
    for raw in values:
        if raw is None:
            continue
        try:
            states.append(RecordingState.model_validate_json(_decode(raw)))
        except ValueError:
            continue
    return states


async def mark_recording_started(
    session_id: str,
    recording_id: str,
    *,
    mode: str,
    interpretation_route: str,
    channels: list[str],
) -> RecordingState:
    state = RecordingState(
        recording_id=recording_id,
        status="recording",
        mode=mode,
        interpretation_route=interpretation_route,
        channels=list(channels),
        started_at=time.time(),
    )
    await _store_state(session_id, state)
    return state


async def mark_recording_stopped(
    session_id: str, recording_id: str, *, summary: dict[str, Any]
) -> RecordingState:
    state = await get_recording_state(session_id, recording_id)
    state = state or RecordingState(recording_id=recording_id, status="stopped")
    state.status = "stopped"
    state.stopped_at = time.time()
    state.summary = dict(summary)
    await _store_state(session_id, state)
    return state


async def mark_recording_reviewed(
    session_id: str, recording_id: str, *, step_count: int
) -> RecordingState:
    state = await get_recording_state(session_id, recording_id)
    state = state or RecordingState(recording_id=recording_id, status="reviewed")
    state.status = "reviewed"
    state.reviewed_at = time.time()
    state.step_count = step_count
    await _store_state(session_id, state)
    return state


async def _store_state(session_id: str, state: RecordingState) -> None:
    redis: Any = await get_redis_async()
    state_key = _recordings_key(session_id)
    order_key = _recording_order_key(session_id)
    await redis.eval(
        _STORE_STATE_SCRIPT,
        2,
        state_key,
        order_key,
        state.recording_id,
        state.model_dump_json(),
        str(time.time()),
        str(_MAX_RECORDINGS_PER_SESSION),
        str(_STATE_TTL_SECONDS),
    )
