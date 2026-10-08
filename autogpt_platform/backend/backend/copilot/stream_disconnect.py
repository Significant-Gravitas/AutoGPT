"""Cluster-wide "stop following this session" for SSE stream listeners.

``DELETE /sessions/{id}/stream`` asks the backend to drop its XREAD listeners
for a session. Listeners live in the memory of whichever API pod serves the
SSE, which is usually not the pod that takes the DELETE, so the request is
broadcast over Redis sharded pub/sub and every pod cancels its own.

A pod subscribes once, lazily, when it starts its first listener: a pod that
never served a stream has nothing to cancel. The broadcast carries the time it
was sent, and only listeners that already existed then are cancelled, so a
user who switches away and straight back keeps the stream they just opened.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus

logger = logging.getLogger(__name__)

# Sharded pub/sub has no pattern-subscribe and a pod cannot know which
# sessions it will serve, so every request goes on one channel.
LISTENER_DISCONNECT_CHANNEL = "all"


class ListenerDisconnect(BaseModel):
    session_id: str
    # Wall clock; listeners started after this are left alone.
    sent_at: float


class ListenerDisconnectBus(AsyncRedisEventBus[ListenerDisconnect]):
    Model = ListenerDisconnect

    @property
    def event_bus_name(self) -> str:
        return "copilot_stream_disconnect"


CancelLocal = Callable[[str, float], Awaitable[int]]


class ListenerDisconnects:
    """One pod's end of the broadcast: publishes requests, and cancels this
    pod's listeners through ``cancel_local(session_id, sent_at)`` for every
    request any pod publishes."""

    def __init__(
        self,
        cancel_local: CancelLocal,
        *,
        bus: ListenerDisconnectBus | None = None,
        retry_delay_s: float = 5.0,
    ) -> None:
        self._cancel_local = cancel_local
        self._bus = bus or ListenerDisconnectBus()
        self._retry_delay_s = retry_delay_s
        self._task: asyncio.Task[None] | None = None

    def ensure_subscribed(self) -> None:
        """Start following the channel on the running loop, once."""
        running = self._task
        if (
            running is not None
            and not running.done()
            and running.get_loop() is asyncio.get_running_loop()
        ):
            return
        self._task = asyncio.create_task(
            self._follow(), name="copilot-stream-disconnects"
        )

    async def broadcast(self, session_id: str) -> None:
        """Ask every pod to cancel the listeners it has for ``session_id``.
        Never raises: a lost request leaves those listeners to end with
        their turn, as they did before this existed."""
        await self._bus.publish_event(
            ListenerDisconnect(session_id=session_id, sent_at=time.time()),
            LISTENER_DISCONNECT_CHANNEL,
        )

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _follow(self) -> None:
        while True:
            try:
                async for request in self._bus.listen_events(
                    LISTENER_DISCONNECT_CHANNEL
                ):
                    await self._apply(request)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Stream disconnect subscription failed: {e}")
            await asyncio.sleep(self._retry_delay_s)

    async def _apply(self, request: ListenerDisconnect) -> None:
        try:
            await self._cancel_local(request.session_id, request.sent_at)
        except Exception as e:
            logger.warning(
                f"Could not cancel listeners of session {request.session_id}: {e}"
            )
