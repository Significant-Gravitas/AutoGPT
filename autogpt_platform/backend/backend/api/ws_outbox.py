"""Events on their way to one WebSocket client.

A socket's shard streams read Redis on their own tasks and hand each event to
the socket's outbox without waiting, so they keep reading Redis, subscribe
confirmations included, however slowly the client reads (#16916). One task
sends the events, in the order they came. A client that falls
``MAX_QUEUED_EVENTS`` events or ``MAX_QUEUED_BYTES`` behind is too slow to
catch up: its events are dropped from then on, and ``on_overflow`` closes the
socket, so the client reconnects, resubscribes and refetches what it missed.
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

# Past either, a client has fallen too far behind to catch up. The bytes are
# what Redis held for a slow subscriber before (its 32 MB pub/sub output
# limit), and two of the event bus's largest messages. A client that keeps up
# stays far under both: a stream reads at most a few hundred KB from Redis
# before the sender runs again.
MAX_QUEUED_EVENTS = 2000
MAX_QUEUED_BYTES = 32 * 1024 * 1024

Delivery = Callable[[], Awaitable[None]]


class WSOutbox:
    def __init__(self, on_overflow: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._on_overflow = on_overflow
        # Deliveries waiting, with their sizes; the first is being sent.
        self._queue: deque[tuple[Delivery, int]] = deque()
        self._queued_bytes = 0
        self._sender: asyncio.Task[None] | None = None
        self._closer: asyncio.Task[None] | None = None
        self._open = True

    def put(self, delivery: Delivery, size: int) -> None:
        """Run ``delivery``, an event holding ``size`` bytes of memory, after
        those put before it. Never waits. ``delivery`` must not raise."""
        if not self._open:
            return
        if (
            len(self._queue) >= MAX_QUEUED_EVENTS
            or self._queued_bytes >= MAX_QUEUED_BYTES
        ):
            self._shut()
            self._closer = asyncio.create_task(self._on_overflow())
            return
        self._queue.append((delivery, size))
        self._queued_bytes += size
        if self._sender is None:
            self._sender = asyncio.create_task(self._send_queued())

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
        self._queued_bytes = 0
        if self._sender is not None:
            # It may be waiting on the client; nothing queued matters now. The
            # server writes each frame whole, so no frame is left half sent.
            self._sender.cancel()

    async def _send_queued(self) -> None:
        while self._queue:
            delivery, size = self._queue[0]
            await delivery()
            if not self._open:
                return
            self._queue.popleft()
            self._queued_bytes -= size
        self._sender = None
