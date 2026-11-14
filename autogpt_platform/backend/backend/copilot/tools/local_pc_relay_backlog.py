"""Bounded appends to the Local PC relay's Redis streams, and their removal.

Each stream holds at most so many entries and bytes. The reply streams, which
the shim's frames fill, also share their connection's reply budget, so a
companion that answers into streams nobody reads any more can leave at most
``MAX_CONNECTION_RESPONSE_BYTES`` behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .local_pc_relay_protocol import RelayBacklogExceeded

STREAM_TTL_SECONDS = 180
MAX_RESPONSE_BACKLOG_ENTRIES = 128
MAX_RESPONSE_BACKLOG_BYTES = 64 * 1024 * 1024
# What all of one connection's reply streams may hold: sixteen frames of
# MAX_ENVELOPE_BYTES, or four full streams. Workers acknowledge answers as they
# read them, so only answers nobody waits for any more build up, until their
# streams expire.
MAX_CONNECTION_RESPONSE_BYTES = 256 * 1024 * 1024
MAX_REQUEST_BACKLOG_ENTRIES = 128
MAX_REQUEST_BACKLOG_BYTES = 64 * 1024 * 1024
_CONNECTION_BUDGET_FULL = -1

# KEYS[1] is a stream and KEYS[2] its byte counter. A reply stream also passes
# its connection's reply budget (reply_budget_keys): KEYS[3] the total, KEYS[4]
# each stream's share of it, set from the stream's own counter on every append
# and lowered with it on every acknowledgement, and KEYS[5] when each stream
# expires.
_RELEASE_REPLY_BYTES = """
local function release_reply_bytes(bytes)
    local held = tonumber(redis.call('GET', KEYS[3]) or '0')
    local released = math.min(bytes, held)
    if released > 0 then
        redis.call('DECRBY', KEYS[3], released)
    end
    return held - released
end
"""

_BOUNDED_APPEND_SCRIPT = (
    _RELEASE_REPLY_BYTES
    + """
-- LOCAL_EXECUTOR_BOUNDED_APPEND
local size = tonumber(ARGV[1])
local stream_bytes = tonumber(redis.call('GET', KEYS[2]) or '0')
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[2])
    or stream_bytes + size > tonumber(ARGV[3]) then
    return 0
end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local expires_at = now + tonumber(ARGV[4]) * 1000
local budget_expires_at = expires_at
if #KEYS == 5 then
    -- Redis tells nobody when a stream expires, so each append takes back
    -- the shares of the streams whose time has passed.
    local freed = 0
    for _, stream in ipairs(redis.call('ZRANGE', KEYS[5], '-inf', now, 'BYSCORE')) do
        freed = freed + tonumber(redis.call('HGET', KEYS[4], stream) or '0')
        redis.call('HDEL', KEYS[4], stream)
    end
    redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', now)
    local share = tonumber(redis.call('HGET', KEYS[4], KEYS[1]) or '0')
    local others = math.max(0, release_reply_bytes(freed) - share)
    if others + stream_bytes + size > tonumber(ARGV[5]) then
        return -1
    end
    redis.call('SET', KEYS[3], others + stream_bytes + size)
    redis.call('HSET', KEYS[4], KEYS[1], stream_bytes + size)
    redis.call('ZADD', KEYS[5], expires_at, KEYS[1])
    -- The budget lives as long as the last of its streams.
    local last = redis.call('ZRANGE', KEYS[5], -1, -1, 'WITHSCORES')
    budget_expires_at = tonumber(last[2])
end
local fields = {}
for index = 6, #ARGV do
    table.insert(fields, ARGV[index])
end
redis.call('XADD', KEYS[1], '*', unpack(fields))
redis.call('INCRBY', KEYS[2], size)
redis.call('PEXPIREAT', KEYS[1], expires_at)
redis.call('PEXPIREAT', KEYS[2], expires_at)
for index = 3, #KEYS do
    redis.call('PEXPIREAT', KEYS[index], budget_expires_at)
end
return 1
"""
)

_ACKNOWLEDGE_SCRIPT = (
    _RELEASE_REPLY_BYTES
    + """
-- LOCAL_EXECUTOR_ACKNOWLEDGE
local removed = redis.call('XDEL', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
local current_bytes = tonumber(redis.call('GET', KEYS[2]) or '0')
local remaining = math.max(0, current_bytes - tonumber(ARGV[2]))
redis.call('SET', KEYS[2], remaining, 'EX', tonumber(ARGV[3]))
if #KEYS == 5 then
    local share = tonumber(redis.call('HGET', KEYS[4], KEYS[1]) or '0')
    if share > remaining then
        release_reply_bytes(share - remaining)
        if remaining > 0 then
            redis.call('HSET', KEYS[4], KEYS[1], remaining)
        else
            redis.call('HDEL', KEYS[4], KEYS[1])
            redis.call('ZREM', KEYS[5], KEYS[1])
        end
    end
end
return 1
"""
)


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


async def append_response(
    redis: Any, key: str, envelope: str, *, budget_keys: Sequence[str]
) -> None:
    """Append a shim frame to a reply stream, within the stream's own limits
    and its connection's reply budget, ``reply_budget_keys``."""
    await _append_bounded_envelope(
        redis,
        key,
        envelope,
        max_entries=MAX_RESPONSE_BACKLOG_ENTRIES,
        max_bytes=MAX_RESPONSE_BACKLOG_BYTES,
        budget_keys=budget_keys,
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
    budget_keys: Sequence[str] = (),
    extra_fields: dict[str, str] | None = None,
) -> None:
    size = len(envelope.encode())
    fields = {"envelope": envelope, "size": str(size)}
    fields.update(extra_fields or {})
    flattened_fields = [item for pair in fields.items() for item in pair]
    keys = [key, _stream_bytes_key(key), *budget_keys]
    appended = await redis.eval(
        _BOUNDED_APPEND_SCRIPT,
        len(keys),
        *keys,
        str(size),
        str(max_entries),
        str(max_bytes),
        str(STREAM_TTL_SECONDS),
        str(MAX_CONNECTION_RESPONSE_BYTES),
        *flattened_fields,
    )
    if int(appended or 0) == _CONNECTION_BUDGET_FULL:
        raise RelayBacklogExceeded(
            "Local executor connection's reply streams exceeded their relay limit"
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
    *,
    budget_keys: Sequence[str] = (),
) -> None:
    """Remove an entry its reader has handled, and give back its bytes: to
    the stream, and with ``budget_keys`` to its connection's reply budget."""
    size = _response_size(fields)
    if size is None:
        await delete_stream_entry(redis, key, message_id)
        return
    keys = [key, _stream_bytes_key(key), *budget_keys]
    await redis.eval(
        _ACKNOWLEDGE_SCRIPT,
        len(keys),
        *keys,
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
