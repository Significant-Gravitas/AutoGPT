"""Worker-side text transport over connection-scoped Redis streams."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol

from .local_pc_relay_protocol import (
    PLATFORM_REQUEST_TYPES,
    RelayBacklogExceeded,
    RelayConnectionReplaced,
    RelayPresence,
    decode_keyed_stream_entries,
    response_stream_key,
    status_stream_key,
    stream_key,
    validate_envelope,
)

STREAM_TTL_SECONDS = 180
POLL_BLOCK_MS = 1_000
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


class RelayTransportBackend(Protocol):
    async def redis(self) -> Any: ...

    async def is_current(self, presence: RelayPresence) -> bool: ...


class RedisRelayTransport:
    def __init__(
        self,
        relay: RelayTransportBackend,
        presence: RelayPresence,
        reply_id: str | None = None,
    ) -> None:
        self._relay = relay
        self._presence = presence
        self._reply_id = reply_id or uuid.uuid4().hex
        self._response_cursor = "0-0"
        # From the start, so a transport opened while the shim is full learns
        # it from the connection's latest STATUS.
        self._status_cursor = "0-0"
        self._closed = asyncio.Event()

    async def send_text(self, data: str) -> None:
        if self._closed.is_set():
            raise ConnectionError("Local executor relay transport is closed")
        validate_envelope(data, PLATFORM_REQUEST_TYPES)
        if not await self._relay.is_current(self._presence):
            raise RelayConnectionReplaced(
                "Local executor connection is no longer active"
            )
        redis = await self._relay.redis()
        key = stream_key(
            self._presence.session_id, self._presence.connection_id, "requests"
        )
        await append_request(redis, key, data, reply_id=self._reply_id)

    async def iter_text(self) -> AsyncIterator[str]:
        """This transport's responses, and the connection's STATUS frames."""
        redis = await self._relay.redis()
        key = response_stream_key(
            self._presence.session_id,
            self._presence.connection_id,
            self._reply_id,
        )
        status_key = status_stream_key(
            self._presence.session_id, self._presence.connection_id
        )
        while not self._closed.is_set() and await self._relay.is_current(
            self._presence
        ):
            # Both keys carry the connection's hash tag, so one XREAD can read
            # them on a cluster.
            result = await redis.xread(
                streams={key: self._response_cursor, status_key: self._status_cursor},
                block=POLL_BLOCK_MS,
                count=100,
            )
            for stream, message_id, fields in _in_stored_order(result):
                envelope = fields.get("envelope")
                if stream == status_key:
                    # Every transport on the connection reads this entry, so
                    # none acknowledges it; the next STATUS replaces it.
                    self._status_cursor = message_id
                    if envelope is not None:
                        yield envelope
                    continue
                self._response_cursor = message_id
                if envelope is None:
                    await acknowledge_stream_entry(redis, key, message_id, fields)
                    continue
                try:
                    yield envelope
                finally:
                    await acknowledge_stream_entry(redis, key, message_id, fields)

    async def close(self) -> None:
        self._closed.set()


def _in_stored_order(result: Any) -> list[tuple[str, str, dict[str, str]]]:
    """One XREAD's entries, oldest first across its streams.

    A response and a STATUS each carry the shim's free slots, and the later
    one should be read last. The streams of one connection share a hash slot,
    so the milliseconds of their entry ids come from one server's clock and
    order them. Each stream numbers its own entries within a millisecond, so
    two written in the same millisecond tie and keep XREAD's order.
    """
    return sorted(
        decode_keyed_stream_entries(result),
        key=lambda entry: _entry_id_order(entry[1]),
    )


def _entry_id_order(entry_id: str) -> tuple[int, int]:
    milliseconds, _, sequence = entry_id.partition("-")
    return int(milliseconds), int(sequence or 0)


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
