"""Events on their way to one WebSocket client.

A socket's shard streams read Redis on their own tasks and hand each event to
the socket's outbox without waiting, so they keep reading Redis, subscribe
confirmations included, however slowly the client reads (#16916). One task
sends the events, in the order they came.

A client too far behind to catch up is closed, so it reconnects, resubscribes
and refetches what it missed; its events are dropped from then on. Too far
behind is ``MAX_QUEUED_EVENTS`` events waiting, or more than
``MAX_QUEUED_BYTES`` of them. When every socket's backlog in the process
together would pass ``MAX_QUEUED_BYTES_PER_PROCESS``, the clients furthest
behind are closed first.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from backend.monitoring.instrumentation import update_websocket_queued_bytes

logger = logging.getLogger(__name__)

# A client that keeps up stays far under both: a stream reads at most a few
# hundred KB from Redis before the sender runs again. MAX_QUEUED_BYTES matches
# Redis's 32 MB pub/sub client-output-buffer limit. Both count the event being
# sent, which is still in memory, so a client behind one very large event is
# closed if the next would pass the bound; two of the event bus's largest (16
# MB) still fit.
MAX_QUEUED_EVENTS = 2000
MAX_QUEUED_BYTES = 32 * 1024 * 1024
# What a queued entry holds beyond its payload: the delivery and its place in
# the queue, measured at about 365 bytes with tracemalloc. Counted with each
# payload, so the byte bounds track the memory an outbox really holds.
ENTRY_OVERHEAD_BYTES = 400
# Each socket holds a shard stream, so the shard-stream cap is all that bounds
# sockets per process: hundreds per shard, which at the per-socket bound alone
# could reach tens of GB.
MAX_QUEUED_BYTES_PER_PROCESS = 1024 * 1024 * 1024

Delivery = Callable[[], Awaitable[None]]
# Called with the bound the client passed and how far behind it was.
OverflowHandler = Callable[[str], Coroutine[Any, Any, None]]


class WSOutbox:
    def __init__(self, on_overflow: OverflowHandler, backlog: "WSBacklog") -> None:
        self._on_overflow = on_overflow
        self._backlog = backlog
        # Deliveries waiting, with their sizes; the first is being sent.
        self._queue: deque[tuple[Delivery, int]] = deque()
        self.queued_bytes = 0
        self._sender: asyncio.Task[None] | None = None
        self._closer: asyncio.Task[None] | None = None
        self._open = True

    @property
    def is_open(self) -> bool:
        """False once the client fell behind, or the socket closed."""
        return self._open

    def put(self, delivery: Delivery, size: int) -> None:
        """Run ``delivery``, an event holding ``size`` bytes of memory, after
        those put before it. Never waits. A delivery that raises is logged
        and skipped."""
        if not self._open:
            return
        if len(self._queue) >= MAX_QUEUED_EVENTS:
            self.overflow("MAX_QUEUED_EVENTS")
            return
        # A client with nothing waiting isn't behind, however big the event.
        if self._queue and self.queued_bytes + size > MAX_QUEUED_BYTES:
            self.overflow("MAX_QUEUED_BYTES")
            return
        if not self._backlog.make_room(size, self):
            return
        self._queue.append((delivery, size))
        self.queued_bytes += size
        self._backlog.changed(self, size)
        if self._sender is None:
            self._sender = asyncio.create_task(self._send_queued())

    def overflow(self, bound: str) -> None:
        """The client passed ``bound``: drop its events and close it. Called
        by the outbox itself and by ``WSBacklog``."""
        if not self._open:
            return
        why = f"{bound}: {len(self._queue)} events, {self.queued_bytes} bytes waiting"
        self._shut()
        self._closer = asyncio.create_task(self._on_overflow(why))

    async def stop(self) -> None:
        """Drop what is queued and end the outbox's tasks."""
        self._shut()
        if self._closer is not None:
            self._closer.cancel()
        for task in (self._sender, self._closer):
            if task is not None:
                await asyncio.wait([task])

    def _shut(self) -> None:
        self._open = False
        self._queue.clear()
        released, self.queued_bytes = self.queued_bytes, 0
        self._backlog.changed(self, -released)
        if self._sender is not None:
            # It may be waiting on the client; nothing queued matters now. The
            # server writes each frame whole, so no frame is left half sent.
            self._sender.cancel()

    async def _send_queued(self) -> None:
        while self._queue:
            delivery, size = self._queue[0]
            try:
                await delivery()
            except Exception:
                # Left to end the task, it would leave the socket's later
                # events waiting on a sender that never runs again.
                logger.exception("Failed to send an event to a WebSocket")
            if not self._open:
                return
            self._queue.popleft()
            self.queued_bytes -= size
            self._backlog.changed(self, -size)
        self._sender = None


class WSBacklog:
    """What one process's outboxes hold together."""

    def __init__(self) -> None:
        self.queued_bytes = 0
        # Outboxes with events waiting.
        self._behind: set[WSOutbox] = set()

    def make_room(self, size: int, outbox: WSOutbox) -> bool:
        """Close the clients furthest behind, most bytes first, until ``size``
        more bytes fit; False if that closed ``outbox``'s own client. An outbox
        with nothing waiting frees nothing, so it is never closed for room;
        each pass takes one outbox out for good, so this ends."""
        if size > MAX_QUEUED_BYTES_PER_PROCESS:
            # Nothing closed would make it fit: only its own client goes.
            outbox.overflow(f"MAX_QUEUED_BYTES_PER_PROCESS, one event of {size} bytes")
            return False
        while self.queued_bytes + size > MAX_QUEUED_BYTES_PER_PROCESS and self._behind:
            furthest = max(self._behind, key=lambda o: o.queued_bytes)
            self._behind.discard(furthest)
            furthest.overflow("MAX_QUEUED_BYTES_PER_PROCESS")
            if furthest is outbox:
                return False
        return True

    def changed(self, outbox: WSOutbox, by: int) -> None:
        """``outbox`` now holds ``by`` bytes more (or fewer)."""
        self.queued_bytes += by
        update_websocket_queued_bytes(by)
        if outbox.queued_bytes:
            self._behind.add(outbox)
        else:
            self._behind.discard(outbox)
