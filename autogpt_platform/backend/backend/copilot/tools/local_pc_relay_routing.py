"""Per-worker response routing for the Local PC Redis relay."""

from __future__ import annotations

import json
import logging
from time import monotonic, time
from typing import Any

from .local_pc_relay_backlog import append_response, append_stream_envelope
from .local_pc_relay_protocol import (
    MAX_STATUS_BYTES,
    NONTERMINAL_RESPONSE_TYPES,
    PLATFORM_REQUEST_TYPES,
    RelayBacklogExceeded,
    RelayRequestTarget,
    is_reply_id,
    register_pending_request,
    reply_budget_keys,
    response_stream_key,
    status_stream_key,
    validate_envelope,
    validate_response_correlation,
)

_REQUEST_CORRELATION_SECONDS = 15 * 60.0
_MAX_PENDING_REQUESTS = 256
_MAX_RECORDING_OWNERS = 32

logger = logging.getLogger(__name__)


class RelayResponseRouter:
    def __init__(self, redis: Any, *, session_id: str, connection_id: str) -> None:
        self._redis = redis
        self._session_id = session_id
        self._connection_id = connection_id
        self._budget_keys = reply_budget_keys(session_id, connection_id)
        self._pending: dict[str, RelayRequestTarget] = {}
        self._pending_activity: dict[str, float] = {}
        self._recording_owners: dict[str, str] = {}
        self._dropped_status_frames = 0
        # Streaming requests that lost a chunk to a full reply stream: the rest
        # of their stream is dropped too, so the worker never reads a gap.
        self._abandoned: set[str] = set()

    async def accept_request(
        self, envelope: str, reply_id: str | None
    ) -> dict[str, Any] | None:
        """Register a worker's request for the machine, or turn it away.

        A request the relay can't take fails alone: it is never sent, and the
        connection carries on for every other request. Ending the connection
        instead would fail them all. Returns the request to send, or None.
        """
        try:
            message = validate_envelope(envelope, PLATFORM_REQUEST_TYPES)
            if reply_id is None or not is_reply_id(reply_id):
                raise ValueError("Local executor request has no valid reply stream")
        except ValueError:
            logger.warning("Dropping a malformed Local PC relay request")
            return None
        try:
            self.register_request(message, reply_id)
        except RelayBacklogExceeded:
            logger.warning(
                "Turning away a Local PC request: "
                f"{_MAX_PENDING_REQUESTS} are already outstanding"
            )
            # An ERROR tells its worker the request was not applied, so it
            # fails now rather than at its own timeout.
            await self._append_or_drop(_overloaded(message["id"]), reply_id)
            return None
        except ValueError:
            # A reused id; the outstanding request keeps its registration.
            logger.warning("Dropping a Local PC request that reused an outstanding id")
            return None
        return message

    def register_request(self, message: dict[str, Any], reply_id: str) -> None:
        self._prune_pending_requests()
        if len(self._pending) >= _MAX_PENDING_REQUESTS:
            raise RelayBacklogExceeded(
                "Local executor has too many outstanding relay requests"
            )
        register_pending_request(message, reply_id, self._pending)
        self._pending_activity[message["id"]] = monotonic()

    def remove_request(self, message_id: str) -> None:
        self._pending.pop(message_id, None)
        self._pending_activity.pop(message_id, None)
        self._abandoned.discard(message_id)

    async def route(self, raw: str, message: dict[str, Any]) -> None:
        self._prune_pending_requests()
        target = validate_response_correlation(message, self._pending)
        if target is not None:
            delivered = await self._deliver(raw, message, target.reply_id)
            # A worker that never saw RECORDING_STARTED does not own it.
            if delivered or message["type"] != "RECORDING_STARTED":
                self._update_recording_owner(message, target)
            return
        if message["type"] == "STATUS":
            await self._publish_status(raw)
            return
        await self._route_recording_step(raw, message)

    async def _deliver(self, raw: str, message: dict[str, Any], reply_id: str) -> bool:
        message_id = message["id"]
        terminal = message["type"] not in NONTERMINAL_RESPONSE_TYPES
        if terminal:
            self._pending_activity.pop(message_id, None)
        else:
            self._pending_activity[message_id] = monotonic()
        if message_id in self._abandoned:
            if terminal:
                self._abandoned.discard(message_id)
            return False
        delivered = await self._append_or_drop(raw, reply_id)
        if not delivered and not terminal:
            self._abandoned.add(message_id)
        return delivered

    async def _append_or_drop(self, raw: str, reply_id: str) -> bool:
        """Append to one worker's reply stream, or drop the frame if that stream
        is full or the connection's reply streams already hold all they may.
        A full stream belongs to a worker that stopped reading, and a full
        budget to a shim that kept answering such workers. Ending the shim
        connection over either would fail every other worker's requests; the
        worker whose frame is dropped times out on its side."""
        key = response_stream_key(self._session_id, self._connection_id, reply_id)
        try:
            await append_response(self._redis, key, raw, budget_keys=self._budget_keys)
        except RelayBacklogExceeded as exc:
            logger.warning(
                f"Dropping a Local PC response for reply stream {reply_id[:8]}: {exc}"
            )
            return False
        return True

    async def _publish_status(self, raw: str) -> None:
        """Keep the shim's latest STATUS once for the whole connection.

        The connection's STATUS stream holds only the newest frame, and every
        transport reads it beside its own reply stream, so a worker holding
        back a request until the shim has a free slot hears about it with
        nothing outstanding. A frame over ``MAX_STATUS_BYTES`` is dropped.
        """
        # The character count is a lower bound on the byte count, so a long
        # frame is refused without encoding it.
        if len(raw) > MAX_STATUS_BYTES or len(raw.encode()) > MAX_STATUS_BYTES:
            self._dropped_status_frames += 1
            # Warn once per connection: a shim that sends one oversized frame
            # sends another every 30s.
            log = logger.warning if self._dropped_status_frames == 1 else logger.debug
            log(
                "Dropping a Local PC STATUS frame over %d bytes (%d dropped on "
                "this connection)",
                MAX_STATUS_BYTES,
                self._dropped_status_frames,
            )
            return
        await append_stream_envelope(
            self._redis,
            status_stream_key(self._session_id, self._connection_id),
            raw,
            maxlen=1,
        )

    async def _route_recording_step(self, raw: str, message: dict[str, Any]) -> None:
        recording_id = message["payload"].get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            logger.warning("Dropping Local PC recording step without a recording id")
            return
        reply_id = self._recording_owners.get(recording_id)
        if reply_id is None:
            logger.warning("Dropping Local PC recording step without an active owner")
            return
        await self._append_or_drop(raw, reply_id)

    def _update_recording_owner(
        self, message: dict[str, Any], target: RelayRequestTarget
    ) -> None:
        recording_id = message["payload"].get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            return
        if (
            target.request_type == "START_RECORDING"
            and message["type"] == "RECORDING_STARTED"
        ):
            if (
                recording_id not in self._recording_owners
                and len(self._recording_owners) >= _MAX_RECORDING_OWNERS
            ):
                oldest = next(iter(self._recording_owners))
                self._recording_owners.pop(oldest, None)
            self._recording_owners[recording_id] = target.reply_id
        elif (
            target.request_type == "STOP_RECORDING"
            and message["type"] == "RECORDING_SUMMARY"
        ):
            self._recording_owners.pop(recording_id, None)

    def _prune_pending_requests(self) -> None:
        cutoff = monotonic() - _REQUEST_CORRELATION_SECONDS
        expired = [
            message_id
            for message_id, last_seen in self._pending_activity.items()
            if last_seen < cutoff
        ]
        for message_id in expired:
            self.remove_request(message_id)


def _overloaded(request_id: str) -> str:
    return json.dumps(
        {
            "type": "ERROR",
            "id": request_id,
            "ts": time(),
            "payload": {
                "code": "SHIM_OVERLOADED",
                "message": (
                    "The Local PC has too many requests in progress. "
                    "Try again in a moment."
                ),
                "details": {"max_concurrent": _MAX_PENDING_REQUESTS},
            },
        },
        separators=(",", ":"),
    )
