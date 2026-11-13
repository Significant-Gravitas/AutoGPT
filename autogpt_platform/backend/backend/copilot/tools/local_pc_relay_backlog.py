"""Bounded appends to the Local PC relay's Redis streams, and their removal."""

from __future__ import annotations

from typing import Any

from .local_pc_relay_protocol import RelayBacklogExceeded

STREAM_TTL_SECONDS = 180
MAX_RESPONSE_BACKLOG_ENTRIES = 128
MAX_RESPONSE_BACKLOG_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BACKLOG_ENTRIES = 128
MAX_REQUEST_BACKLOG_BYTES = 64 * 1024 * 1024

_BOUNDED_APPEND_SCRIPT = """
-- LOCAL_EXECUTOR_BOUNDED_APPEND
local entries = redis.call('XLEN', KEYS[1])
local current_bytes = tonumber(redis.call('GET', KEYS[2]) or '0')
local size = tonumber(ARGV[1])
local max_entries = tonumber(ARGV[2])
local max_bytes = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
if entries >= max_entries or current_bytes + size > max_bytes then
    return 0
end
local fields = {}
for index = 5, #ARGV do
    table.insert(fields, ARGV[index])
end
redis.call('XADD', KEYS[1], '*', unpack(fields))
redis.call('INCRBY', KEYS[2], size)
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[2], ttl)
return 1
"""

_ACKNOWLEDGE_SCRIPT = """
-- LOCAL_EXECUTOR_ACKNOWLEDGE
local removed = redis.call('XDEL', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
local current_bytes = tonumber(redis.call('GET', KEYS[2]) or '0')
local remaining = math.max(0, current_bytes - tonumber(ARGV[2]))
redis.call('SET', KEYS[2], remaining, 'EX', tonumber(ARGV[3]))
return 1
"""


async def append_stream_envelope(
    redis: Any,
    key: str,
    envelope: str,
    *,
    maxlen: int,
    extra_fields: dict[str, str] | None = None,
) -> None:
    fields = {"envelope": envelope}
    fields.update(extra_fields or {})
    pipeline = redis.pipeline(transaction=True)
    pipeline.xadd(
        key,
        fields,
        maxlen=maxlen,
        approximate=False,
    )
    pipeline.expire(key, STREAM_TTL_SECONDS)
    await pipeline.execute()


async def append_response(redis: Any, key: str, envelope: str) -> None:
    await _append_bounded_envelope(
        redis,
        key,
        envelope,
        max_entries=MAX_RESPONSE_BACKLOG_ENTRIES,
        max_bytes=MAX_RESPONSE_BACKLOG_BYTES,
    )


async def append_request(redis: Any, key: str, envelope: str, *, reply_id: str) -> None:
    await _append_bounded_envelope(
        redis,
        key,
        envelope,
        max_entries=MAX_REQUEST_BACKLOG_ENTRIES,
        max_bytes=MAX_REQUEST_BACKLOG_BYTES,
        extra_fields={"reply_id": reply_id},
    )


async def _append_bounded_envelope(
    redis: Any,
    key: str,
    envelope: str,
    *,
    max_entries: int,
    max_bytes: int,
    extra_fields: dict[str, str] | None = None,
) -> None:
    size = len(envelope.encode())
    counter_key = _stream_bytes_key(key)
    fields = {"envelope": envelope, "size": str(size)}
    fields.update(extra_fields or {})
    flattened_fields = [item for pair in fields.items() for item in pair]
    appended = await redis.eval(
        _BOUNDED_APPEND_SCRIPT,
        2,
        key,
        counter_key,
        str(size),
        str(max_entries),
        str(max_bytes),
        str(STREAM_TTL_SECONDS),
        *flattened_fields,
    )
    if int(appended or 0) != 1:
        raise RelayBacklogExceeded(
            "Local executor stream backlog exceeded its relay limit"
        )


async def acknowledge_stream_entry(
    redis: Any,
    key: str,
    message_id: str,
    fields: dict[str, str],
) -> None:
    size = _response_size(fields)
    if size is None:
        await delete_stream_entry(redis, key, message_id)
        return
    counter_key = _stream_bytes_key(key)
    await redis.eval(
        _ACKNOWLEDGE_SCRIPT,
        2,
        key,
        counter_key,
        message_id,
        str(size),
        str(STREAM_TTL_SECONDS),
    )


async def delete_stream_entry(redis: Any, key: str, message_id: str) -> None:
    await redis.xdel(key, message_id)


def _response_size(fields: dict[str, str]) -> int | None:
    try:
        return max(0, int(fields["size"]))
    except KeyError:
        return None
    except ValueError:
        return None


def _stream_bytes_key(stream: str) -> str:
    return f"{stream}:backlog-bytes"
