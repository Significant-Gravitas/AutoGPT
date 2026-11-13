"""Worker-side text transport over connection-scoped Redis streams."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol

from .local_pc_relay_backlog import acknowledge_stream_entry, append_request
from .local_pc_relay_protocol import (
    PLATFORM_REQUEST_TYPES,
    RelayConnectionReplaced,
    RelayPresence,
    decode_keyed_stream_entries,
    reply_budget_keys,
    response_stream_key,
    status_stream_key,
    stream_key,
    validate_envelope,
)

POLL_BLOCK_MS = 1_000


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
        # Reading an answer gives its bytes back to the connection's budget.
        budget_keys = reply_budget_keys(
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
                    await acknowledge_stream_entry(
                        redis, key, message_id, fields, budget_keys=budget_keys
                    )
                    continue
                try:
                    yield envelope
                finally:
                    await acknowledge_stream_entry(
                        redis, key, message_id, fields, budget_keys=budget_keys
                    )

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
