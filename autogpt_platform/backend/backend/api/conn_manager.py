import asyncio
import json
import logging
import sys
from collections import Counter
from functools import partial
from typing import Awaitable, Callable, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.api.model import WSMessage, WSMethod
from backend.api.ws_outbox import ENTRY_OVERHEAD_BYTES, WSBacklog, WSOutbox
from backend.data import redis_client as redis
from backend.data.db_accessors import execution_db
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import ExecutionEventType, exec_channel, graph_all_channel
from backend.data.notification_bus import NotificationEvent
from backend.data.sharded_pubsub import EventStreamLostError, ShardedSubscription
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_settings = Settings()


def _is_ws_close_race(exc: BaseException, websocket: WebSocket) -> bool:
    """A SPUBLISH→WS send racing with WS close — benign, drop quietly."""
    if isinstance(exc, WebSocketDisconnect):
        return True
    if (
        getattr(websocket, "application_state", None) == WebSocketState.DISCONNECTED
        or getattr(websocket, "client_state", None) == WebSocketState.DISCONNECTED
    ):
        return True
    if isinstance(exc, RuntimeError) and "close message has been sent" in str(exc):
        return True
    return False


_EVENT_TYPE_TO_METHOD_MAP: dict[ExecutionEventType, WSMethod] = {
    ExecutionEventType.GRAPH_EXEC_UPDATE: WSMethod.GRAPH_EXECUTION_EVENT,
    ExecutionEventType.NODE_EXEC_UPDATE: WSMethod.NODE_EXECUTION_EVENT,
}


def event_bus_channel(channel_key: str) -> str:
    """Prefix a channel key with the execution event bus name."""
    return f"{_settings.config.execution_event_bus_name}/{channel_key}"


def _notification_bus_channel(user_id: str) -> str:
    """Return the full sharded channel name for a user's notifications."""
    return f"{_settings.config.notification_event_bus_name}/{user_id}"


MessageHandler = Callable[[Optional[bytes | str]], Awaitable[None]]
LostHandler = Callable[[], Awaitable[None]]

# Execution subscriptions one socket may hold. The notification channel each
# socket gets on connect is not counted.
MAX_SUBSCRIPTIONS_PER_SOCKET = 1000
# Sockets one user may hold open on this server process.
MAX_SOCKETS_PER_USER = 32
# Shard streams (Redis connections) this process may hold per shard, over all
# its sockets; while a shard is at it, new sockets are refused.
MAX_STREAMS_PER_SHARD = 300
# Streams one user may open on a shard while it has room; beyond this share a
# user may not take the shard's last quarter, so heavy users can't fill it.
STREAMS_PER_USER_SHARE = 8
# SSUBSCRIBE on an open shard stream has no socket timeout, and Redis's
# confirmation can lag; give up after this so the handler answers and the
# router reads the socket again. Opening the stream is bounded by
# ShardedSubscription.open itself.
_SUBSCRIBE_TIMEOUT_S = 15.0
# Taking back an SSUBSCRIBE whose confirmation timed out is only a write.
_TAKE_BACK_TIMEOUT_S = 2.0

WS_CLOSE_EVENT_STREAM_LOST = 1011
WS_CLOSE_TRY_AGAIN_LATER = 1013

Shard = tuple[str, int]


class _StreamCapacityError(ValueError):
    """A shard had no room left for this user's stream."""


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # channel_key → sockets subscribed (public channel keys, not raw Redis channels)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        # websocket → {channel_key: full Redis channel}
        self._ws_subs: dict[WebSocket, dict[str, str]] = {}
        # websocket → its notification channel
        self._ws_notifications: dict[WebSocket, str] = {}
        # websocket → {shard: the one connection it subscribes over there}
        self._ws_streams: dict[WebSocket, dict[Shard, _ShardStream]] = {}
        # websocket → its events on their way to the client, from every shard
        self._outboxes: dict[WebSocket, WSOutbox] = {}
        # what every socket's outbox holds together
        self._backlog = WSBacklog()
        # user_id → that user's sockets on this process
        self._user_sockets: dict[str, set[WebSocket]] = {}
        # shard → shard streams open to it from this process
        self._streams_per_shard: Counter[Shard] = Counter()
        # (user_id, shard) → that user's streams open to it
        self._user_streams: Counter[tuple[str, Shard]] = Counter()

    async def connect_socket(self, websocket: WebSocket, *, user_id: str) -> bool:
        """Accept the socket, or close it and return False: with 1013 when the
        user holds ``MAX_SOCKETS_PER_USER`` here or a shard has no room left
        for the user's streams, and with 1011 when the notification stream
        can't be opened."""
        held = len(self._user_sockets.get(user_id, ()))
        if held >= MAX_SOCKETS_PER_USER or self._a_shard_is_full(user_id):
            await self._refuse(websocket, user_id)
            return False
        self._user_sockets.setdefault(user_id, set()).add(websocket)
        try:
            await websocket.accept()
        except BaseException:
            self._release_user_socket(user_id, websocket)
            raise
        self.active_connections.add(websocket)
        self._ws_subs.setdefault(websocket, {})
        # Don't hand back a socket that will never receive a notification.
        try:
            await self._start_notification_subscription(websocket, user_id=user_id)
        except _StreamCapacityError:
            # Capacity moved between admission and here.
            await self._turn_away(
                websocket, user_id, WS_CLOSE_TRY_AGAIN_LATER, "Try again later"
            )
            return False
        except Exception:
            # Redis is unreachable, or the stream was lost before Redis
            # confirmed it: closing tells the client to reconnect and
            # resubscribe.
            logger.warning(
                f"Closing WebSocket for user #{user_id}: its notification "
                "stream could not be opened",
                exc_info=True,
            )
            await self._turn_away(
                websocket,
                user_id,
                WS_CLOSE_EVENT_STREAM_LOST,
                "Event stream interrupted",
            )
            return False
        return True

    async def _turn_away(
        self, websocket: WebSocket, user_id: str, code: int, reason: str
    ) -> None:
        try:
            await websocket.close(code=code, reason=reason)
        finally:
            await self.disconnect_socket(websocket, user_id=user_id)

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
        self._release_user_socket(user_id, websocket)
        # Stop the shard streams before dropping bookkeeping to avoid leaks.
        try:
            for shard, stream in self._ws_streams.pop(websocket, {}).items():
                await stream.stop()
                self._release_stream(user_id, shard)
        finally:
            # Even when a stop is cancelled (the server shutting down), so the
            # socket's backlog stops counting against the process.
            outbox = self._outboxes.pop(websocket, None)
            if outbox is not None:
                await outbox.stop()
        self._ws_notifications.pop(websocket, None)
        for channel_key in self._ws_subs.pop(websocket, {}):
            subscribers = self.subscriptions.get(channel_key)
            if subscribers is None:
                continue
            subscribers.discard(websocket)
            if not subscribers:
                self.subscriptions.pop(channel_key, None)

    async def subscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str:
        # Hash-tagged channel needs graph_id; resolve once per subscribe.
        meta = await execution_db().get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if meta is None:
            raise ValueError(
                f"graph_exec #{graph_exec_id} not found for user #{user_id}"
            )
        channel_key = graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        full_channel = event_bus_channel(
            exec_channel(user_id, meta.graph_id, graph_exec_id)
        )
        await self._open_subscription(user_id, websocket, channel_key, full_channel)
        return channel_key

    async def subscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        full_channel = event_bus_channel(graph_all_channel(user_id, graph_id))
        await self._open_subscription(user_id, websocket, channel_key, full_channel)
        return channel_key

    async def unsubscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        return await self._close_subscription(user_id, websocket, channel_key)

    async def unsubscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        return await self._close_subscription(user_id, websocket, channel_key)

    async def _open_subscription(
        self, user_id: str, websocket: WebSocket, channel_key: str, full_channel: str
    ) -> None:
        per_ws = self._ws_subs.setdefault(websocket, {})
        if channel_key not in per_ws:
            if len(per_ws) >= MAX_SUBSCRIPTIONS_PER_SOCKET:
                raise ValueError(
                    f"Subscription limit reached ({MAX_SUBSCRIPTIONS_PER_SOCKET} "
                    "per connection)"
                )

            async def on_message(data: Optional[bytes | str]) -> None:
                await self._forward_exec_event(websocket, channel_key, data)

            await self._subscribe(user_id, websocket, full_channel, on_message)
            per_ws[channel_key] = full_channel
        self.subscriptions.setdefault(channel_key, set()).add(websocket)

    async def _close_subscription(
        self, user_id: str, websocket: WebSocket, channel_key: str
    ) -> str | None:
        subscribers = self.subscriptions.get(channel_key)
        if subscribers is None:
            return None
        subscribers.discard(websocket)
        if not subscribers:
            self.subscriptions.pop(channel_key, None)
        full_channel = self._ws_subs.get(websocket, {}).pop(channel_key, None)
        if full_channel is not None:
            await self._unsubscribe(user_id, websocket, full_channel)
        return channel_key

    async def _subscribe(
        self,
        user_id: str,
        websocket: WebSocket,
        full_channel: str,
        handler: MessageHandler,
    ) -> None:
        """SSUBSCRIBE on the socket's stream for the channel's shard, opening
        that stream if this is the socket's first channel there."""
        outbox = self._outbox_for(user_id, websocket)
        if not outbox.is_open:
            # Retired for falling behind: a stream would read for nothing.
            raise EventStreamLostError(f"{full_channel}: the socket is closing")
        shard = redis.resolve_shard_for_channel(full_channel)
        streams = self._ws_streams.setdefault(websocket, {})
        stream = streams.get(shard)
        if stream is None:
            if not self._has_room(user_id, shard):
                raise _StreamCapacityError("Server busy, try again later")

            async def on_lost() -> None:
                await self._close_for_lost_stream(websocket)

            stream = streams[shard] = _ShardStream(on_lost, outbox)
            self._streams_per_shard[shard] += 1
            self._user_streams[(user_id, shard)] += 1
        try:
            await stream.add(full_channel, handler)
        except BaseException:
            if not stream.handlers and streams.pop(shard, None) is not None:
                self._release_stream(user_id, shard)
            raise

    async def _unsubscribe(
        self, user_id: str, websocket: WebSocket, full_channel: str
    ) -> None:
        streams = self._ws_streams.get(websocket, {})
        for shard, stream in list(streams.items()):
            if full_channel not in stream.handlers:
                continue
            await stream.remove(full_channel)
            if not stream.handlers:
                streams.pop(shard, None)
                self._release_stream(user_id, shard)
            return

    def _outbox_for(self, user_id: str, websocket: WebSocket) -> WSOutbox:
        """The socket's outbox, made on first use; disconnect_socket drops it."""
        outbox = self._outboxes.get(websocket)
        if outbox is None:

            async def on_overflow(why: str) -> None:
                await self._close_for_slow_client(websocket, user_id, why)

            outbox = self._outboxes[websocket] = WSOutbox(on_overflow, self._backlog)
        return outbox

    async def _close_for_lost_stream(self, websocket: WebSocket) -> None:
        """Close a socket whose events stopped arriving, so the client
        reconnects, resubscribes and refetches what it missed."""
        logger.warning("Closing WebSocket after its event stream was lost")
        await self._close_to_resync(websocket, "Event stream interrupted")

    async def _close_for_slow_client(
        self, websocket: WebSocket, user_id: str, why: str
    ) -> None:
        """Close a socket whose client fell too far behind: the events it
        would get next are dropped, so it resyncs as after a lost stream. Its
        outbox shut before this ran, which retires the socket: ``_subscribe``
        refuses it new streams from then on."""
        logger.warning(
            f"Closing WebSocket for user #{user_id}: its client fell too far "
            f"behind ({why})"
        )
        try:
            # Nothing they read would reach the client now, and the close may
            # never get through a transport the client has stopped reading. A
            # copy: the socket's own messages can still drop streams meanwhile.
            for stream in list(self._ws_streams.get(websocket, {}).values()):
                try:
                    await stream.stop()
                except Exception:
                    logger.warning(
                        "Failed to stop a slow client's stream", exc_info=True
                    )
        finally:
            await self._close_to_resync(websocket, "Client fell behind")

    async def _close_to_resync(self, websocket: WebSocket, reason: str) -> None:
        try:
            await websocket.close(code=WS_CLOSE_EVENT_STREAM_LOST, reason=reason)
        except Exception as e:
            if not _is_ws_close_race(e, websocket):
                logger.warning("Failed to close WebSocket", exc_info=True)

    async def _refuse(self, websocket: WebSocket, user_id: str) -> None:
        # Accepted first, so the client gets 1013 rather than a failed handshake.
        logger.warning(f"Refusing WebSocket for user #{user_id}: server busy")
        try:
            await websocket.accept()
            await websocket.close(
                code=WS_CLOSE_TRY_AGAIN_LATER, reason="Try again later"
            )
        except Exception as e:
            if not _is_ws_close_race(e, websocket):
                logger.warning("Failed to refuse WebSocket", exc_info=True)

    def _a_shard_is_full(self, user_id: str) -> bool:
        return any(not self._has_room(user_id, s) for s in self._streams_per_shard)

    def _has_room(self, user_id: str, shard: Shard) -> bool:
        held = self._streams_per_shard[shard]
        if self._user_streams[(user_id, shard)] < STREAMS_PER_USER_SHARE:
            return held < MAX_STREAMS_PER_SHARD
        return held < MAX_STREAMS_PER_SHARD - MAX_STREAMS_PER_SHARD // 4

    def _release_stream(self, user_id: str, shard: Shard) -> None:
        self._streams_per_shard[shard] -= 1
        if self._streams_per_shard[shard] <= 0:
            del self._streams_per_shard[shard]
        self._user_streams[(user_id, shard)] -= 1
        if self._user_streams[(user_id, shard)] <= 0:
            del self._user_streams[(user_id, shard)]

    def _release_user_socket(self, user_id: str, websocket: WebSocket) -> None:
        sockets = self._user_sockets.get(user_id)
        if sockets is None:
            return
        sockets.discard(websocket)
        if not sockets:
            self._user_sockets.pop(user_id, None)

    async def _forward_exec_event(
        self,
        websocket: WebSocket,
        channel_key: str,
        raw_payload: Optional[bytes | str],
    ) -> None:
        if raw_payload is None:
            return
        # Unwrap the `_EventPayloadWrapper` envelope, then re-wrap as a WS message.
        try:
            wrapper = (
                raw_payload.decode()
                if isinstance(raw_payload, (bytes, bytearray))
                else raw_payload
            )
        except Exception:
            logger.warning(
                "Failed to decode pubsub payload on %s", channel_key, exc_info=True
            )
            return

        try:
            parsed = json.loads(wrapper)
            event_data = parsed.get("payload")
            if not isinstance(event_data, dict):
                return
            event_type = event_data.get("event_type")
            method = _EVENT_TYPE_TO_METHOD_MAP.get(ExecutionEventType(event_type))
            if method is None:
                return
            message = WSMessage(
                method=method,
                channel=channel_key,
                data=event_data,
            ).model_dump_json()
            await websocket.send_text(message)
        except Exception as e:
            if _is_ws_close_race(e, websocket):
                logger.debug("Dropped exec event on closed WS for %s", channel_key)
                return
            logger.exception("Failed to forward exec event on %s", channel_key)

    async def _start_notification_subscription(
        self, websocket: WebSocket, *, user_id: str
    ) -> None:
        full_channel = _notification_bus_channel(user_id)

        async def on_message(data: Optional[bytes | str]) -> None:
            await self._forward_notification(websocket, user_id, data)

        await self._subscribe(user_id, websocket, full_channel, on_message)
        self._ws_notifications[websocket] = full_channel

    async def _forward_notification(
        self,
        websocket: WebSocket,
        user_id: str,
        raw_payload: Optional[bytes | str],
    ) -> None:
        if raw_payload is None:
            return
        try:
            wrapper_json = (
                raw_payload.decode()
                if isinstance(raw_payload, (bytes, bytearray))
                else raw_payload
            )
            parsed = json.loads(wrapper_json)
            inner = parsed.get("payload") if isinstance(parsed, dict) else None
            if not isinstance(inner, dict):
                return
            event = NotificationEvent.model_validate(inner)
        except Exception:
            logger.warning(
                "Failed to parse notification payload for user=%s",
                user_id,
                exc_info=True,
            )
            return
        # Defense in depth against cross-user payloads.
        if event.user_id != user_id:
            return
        message = WSMessage(
            method=WSMethod.NOTIFICATION,
            data=event.payload.model_dump(),
        ).model_dump_json()
        try:
            await websocket.send_text(message)
        except Exception as e:
            if _is_ws_close_race(e, websocket):
                logger.debug("Dropped notification on closed WS for user=%s", user_id)
                return
            logger.warning(
                "Failed to deliver notification to WS for user=%s",
                user_id,
                exc_info=True,
            )


class _ShardStream:
    """One socket's SSUBSCRIBEs on one Redis shard, all over one connection.
    Its events go through the socket's outbox, so a client that reads slowly
    never holds up reading Redis."""

    def __init__(self, on_lost: LostHandler, outbox: WSOutbox) -> None:
        # full channel → handler
        self.handlers: dict[str, MessageHandler] = {}
        self._on_lost = on_lost
        self._outbox = outbox
        self._subscription: ShardedSubscription | None = None
        self._task: asyncio.Task | None = None
        # SUNSUBSCRIBEs sent per channel whose confirmation is still on its way
        self._unsubscribing: Counter[str] = Counter()
        # SSUBSCRIBEs sent on the open stream, until Redis confirms them. One
        # per channel: ws_api handles a socket's messages one at a time, and
        # add() refuses a second subscribe of a channel still in here.
        self._subscribing: dict[str, asyncio.Future[None]] = {}
        # Once the pump ends, nothing reads Redis's confirmations any more.
        self._pump_ended = False
        self._stopping = False

    async def add(self, channel: str, handler: MessageHandler) -> None:
        _assert_no_wildcard(channel)
        if self._stopping:
            # Retired with its socket closing; reopened, it would read for a
            # client that is gone.
            raise EventStreamLostError(f"{channel}: the stream has stopped")
        if channel in self._subscribing:
            raise RuntimeError(f"{channel} is already being subscribed")
        # In place before the SSUBSCRIBE: the pump reads a message queued
        # right behind Redis's confirmation without pausing for this.
        self.handlers[channel] = handler
        try:
            subscription = await self._open_or_subscribe(channel)
            if self._stopping or not self._outbox.is_open:
                # Stopped while this opened its connection, so that stop
                # closed nothing, or about to be, the socket having been
                # retired meanwhile: closed below, as no pump will read it.
                raise EventStreamLostError(f"{channel}: the stream has stopped")
        except BaseException:
            self.handlers.pop(channel, None)
            if not self.handlers:
                await self._close_quietly()
            raise
        if self._task is None:
            self._task = asyncio.create_task(self._pump(subscription))

    async def _open_or_subscribe(self, channel: str) -> ShardedSubscription:
        if self._subscription is None:
            # open() bounds itself, then refreshes the slot map on a timeout;
            # a bound of ours would cancel it first.
            self._subscription = await ShardedSubscription.open(channel)
        else:
            await self._subscribe_confirmed(self._subscription, channel)
        return self._subscription

    async def _subscribe_confirmed(
        self, subscription: ShardedSubscription, channel: str
    ) -> None:
        """SSUBSCRIBE and return once Redis confirms it, as ``open`` does for
        the first channel. The socket answers the subscribe after this, and a
        client re-reads its state then: an event Redis handled before the
        SSUBSCRIBE would never arrive. The pump reads the confirmation."""
        if self._pump_ended:
            # The socket is already closing for the lost stream.
            raise EventStreamLostError(f"{channel}: the stream has ended")
        confirmed = asyncio.get_running_loop().create_future()
        self._subscribing[channel] = confirmed
        try:
            # SSUBSCRIBE has no socket timeout of its own.
            async with _subscribe_deadline():
                await subscription.add(channel)
                await confirmed
        except TimeoutError:
            await self._take_back(subscription, channel)
            raise
        finally:
            if self._subscribing.get(channel) is confirmed:
                del self._subscribing[channel]
            # The pump may have failed it before the write failed too.
            if confirmed.done() and not confirmed.cancelled():
                confirmed.exception()

    async def _take_back(self, subscription: ShardedSubscription, channel: str) -> None:
        """SUNSUBSCRIBE a channel whose confirmation timed out: Redis may have
        applied the SSUBSCRIBE anyway, and would then keep sending it with no
        handler to take it until the stream closes. If even that write fails,
        retire the stream, so the socket closes and the client resubscribes."""
        if self._pump_ended:
            # The socket is already closing for the lost stream.
            return
        self._unsubscribing[channel] += 1
        try:
            async with asyncio.timeout(_TAKE_BACK_TIMEOUT_S):
                await subscription.remove(channel)
        except Exception:
            # Any failure: no confirmation is coming, and a stale count would
            # hide the channel's events and a real drop.
            self._unsubscribed(channel)
            logger.warning(f"Could not take back {channel}", exc_info=True)
            if not self._pump_ended:
                await self.stop()
                await self._on_lost()

    async def remove(self, channel: str) -> None:
        if self.handlers.pop(channel, None) is None:
            return
        if not self.handlers:
            await self.stop()
            return
        if self._subscription is None:
            return
        self._unsubscribing[channel] += 1
        try:
            await self._subscription.remove(channel)
        except Exception:
            # No confirmation is coming; a stale count would hide a real drop.
            self._unsubscribed(channel)
            logger.warning(f"SUNSUBSCRIBE failed for {channel}", exc_info=True)

    def _unsubscribed(self, channel: str) -> None:
        """One SUNSUBSCRIBE of the channel is answered, or won't be."""
        self._unsubscribing[channel] -= 1
        if self._unsubscribing[channel] <= 0:
            del self._unsubscribing[channel]

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # Closing the connection drops every SSUBSCRIBE it carried.
        await self._close_quietly()

    async def _close_quietly(self) -> None:
        if self._subscription is not None:
            await self._subscription.aclose()
            self._subscription = None

    async def _pump(self, subscription: ShardedSubscription) -> None:
        try:
            await self._forward(subscription)
        except asyncio.CancelledError:
            raise
        except EventStreamLostError as e:
            logger.warning(f"Pubsub stream lost: {e}")
        except Exception:
            logger.exception("Pubsub pump crashed")
        finally:
            # No confirmation can come once the stream is gone.
            self._pump_ended = True
            for confirmed in self._subscribing.values():
                if not confirmed.done():
                    confirmed.set_exception(
                        EventStreamLostError("the stream ended before Redis confirmed")
                    )
        if not self._stopping:
            await self._on_lost()

    async def _forward(self, subscription: ShardedSubscription) -> None:
        async for message in subscription.messages():
            msg_type = message["type"]
            channel = message["channel"]
            if msg_type == "sunsubscribe":
                if self._unsubscribing[channel] > 0:
                    self._unsubscribed(channel)
                    continue
                # Not ours: the slot moved and Redis dropped the subscription.
                return
            if msg_type == "ssubscribe":
                # Redis answers in order, so while a SUNSUBSCRIBE of the
                # channel is unanswered this confirms an SSUBSCRIBE sent
                # before it, which nothing waits on any more.
                confirmed = self._subscribing.get(channel)
                if (
                    self._unsubscribing[channel] == 0
                    and confirmed is not None
                    and not confirmed.done()
                ):
                    confirmed.set_result(None)
                continue
            if msg_type != "smessage":
                continue
            # Published before an unanswered SUNSUBSCRIBE: it belongs to the
            # subscription being dropped, not to one made since.
            if self._unsubscribing[channel] > 0:
                continue
            handler = self.handlers.get(channel) if channel else None
            if handler is None:
                continue
            # Never awaited here: a send waiting on a slow client would hold
            # up the next read, a subscribe's confirmation included.
            data = message.get("data")
            self._outbox.put(
                partial(self._deliver, channel, handler, data),
                sys.getsizeof(data) + ENTRY_OVERHEAD_BYTES,
            )

    async def _deliver(
        self, channel: str, handler: MessageHandler, data: Optional[bytes | str]
    ) -> None:
        # Unsubscribed, or subscribed again, while it waited: it belongs to a
        # subscription the client has let go of.
        if self.handlers.get(channel) is not handler:
            return
        try:
            await handler(data)
        except Exception:
            logger.exception(f"Websocket message-handler failed for {channel}")


def _subscribe_deadline() -> asyncio.Timeout:
    """The bound on Redis confirming one SSUBSCRIBE, from now: a function of
    its own so a test can expire it at the point it chooses."""
    return asyncio.timeout(_SUBSCRIBE_TIMEOUT_S)


def graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
