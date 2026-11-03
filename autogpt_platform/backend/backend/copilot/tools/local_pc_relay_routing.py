"""Per-worker response routing for the Local PC Redis relay."""

from __future__ import annotations

import logging
from time import monotonic
from typing import Any

from .local_pc_relay_protocol import (
    NONTERMINAL_RESPONSE_TYPES,
    RelayBacklogExceeded,
    RelayRequestTarget,
    register_pending_request,
    response_stream_key,
    validate_response_correlation,
)
from .local_pc_relay_transport import append_response

_REPLY_ACTIVITY_SECONDS = 300.0
_REQUEST_CORRELATION_SECONDS = 15 * 60.0
_MAX_PENDING_REQUESTS = 256
_MAX_RECORDING_OWNERS = 32

logger = logging.getLogger(__name__)


class RelayResponseRouter:
    def __init__(self, redis: Any, *, session_id: str, connection_id: str) -> None:
        self._redis = redis
        self._session_id = session_id
        self._connection_id = connection_id
        self._pending: dict[str, RelayRequestTarget] = {}
        self._pending_activity: dict[str, float] = {}
        self._reply_activity: dict[str, float] = {}
        self._recording_owners: dict[str, str] = {}
        # Streaming requests that lost a chunk to a full reply stream: the rest
        # of their stream is dropped too, so the worker never reads a gap.
        self._abandoned: set[str] = set()

    def register_request(self, message: dict[str, Any], reply_id: str) -> None:
        self._prune_pending_requests()
        if len(self._pending) >= _MAX_PENDING_REQUESTS:
            raise RelayBacklogExceeded(
                "Local executor has too many outstanding relay requests"
            )
        target = register_pending_request(message, reply_id, self._pending)
        now = monotonic()
        self._pending_activity[message["id"]] = now
        self._reply_activity[target.reply_id] = now

    def remove_request(self, message_id: str) -> None:
        self._pending.pop(message_id, None)
        self._pending_activity.pop(message_id, None)
        self._abandoned.discard(message_id)

    async def route(self, raw: str, message: dict[str, Any]) -> None:
        self._prune_pending_requests()
        target = validate_response_correlation(message, self._pending)
        if target is not None:
            await self._deliver(raw, message, target.reply_id)
            self._update_recording_owner(message, target)
            return
        if message["type"] == "STATUS":
            await self._broadcast_status(raw)
            return
        await self._route_recording_step(raw, message)

    async def _deliver(self, raw: str, message: dict[str, Any], reply_id: str) -> None:
        message_id = message["id"]
        terminal = message["type"] not in NONTERMINAL_RESPONSE_TYPES
        if terminal:
            self._pending_activity.pop(message_id, None)
        else:
            self._pending_activity[message_id] = monotonic()
        if message_id in self._abandoned:
            if terminal:
                self._abandoned.discard(message_id)
            return
        if not await self._append_or_drop(raw, reply_id) and not terminal:
            self._abandoned.add(message_id)

    async def _append_or_drop(self, raw: str, reply_id: str) -> bool:
        """Append to one worker's reply stream, or drop the frame if that stream
        is full. A full stream belongs to a worker that stopped reading; ending
        the shim connection over it would fail every other worker's requests,
        and the stalled worker's own request times out on its side."""
        key = response_stream_key(self._session_id, self._connection_id, reply_id)
        try:
            await append_response(self._redis, key, raw)
        except RelayBacklogExceeded:
            self._reply_activity.pop(reply_id, None)
            logger.warning(
                "Dropping a Local PC response for a full reply stream %s",
                reply_id[:8],
            )
            return False
        return True

    async def _broadcast_status(self, raw: str) -> None:
        cutoff = monotonic() - _REPLY_ACTIVITY_SECONDS
        active = {
            reply_id
            for reply_id, last_seen in self._reply_activity.items()
            if last_seen >= cutoff
        }
        active.update(target.reply_id for target in self._pending.values())
        active.update(self._recording_owners.values())
        self._reply_activity = {
            reply_id: last_seen
            for reply_id, last_seen in self._reply_activity.items()
            if reply_id in active
        }
        # A full stream drops out of _reply_activity, so a worker that stopped
        # reading and has nothing pending stops getting STATUS.
        for reply_id in active:
            await self._append_or_drop(raw, reply_id)

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
