import asyncio
import json
import logging
from typing import Awaitable, Callable, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.exceptions import RedisError
from starlette.websockets import WebSocketState

from backend.api.model import WSMessage, WSMethod
from backend.data import redis_client as redis
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import (
    ExecutionEventType,
    exec_channel,
    get_graph_execution_meta,
    graph_all_channel,
)
from backend.data.notification_bus import NotificationEvent
from backend.monitoring.instrumentation import record_websocket_event_stream_lost
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
# Called once with a short cause when a subscription's event stream ends.
LostHandler = Callable[[str], Awaitable[None]]

# Opening a shard-pinned SSUBSCRIBE retries for minutes while Redis is down;
# give up sooner so the socket is closed and the client retries with backoff.
_SUBSCRIBE_OPEN_TIMEOUT_S = 15.0

# Close code for a socket whose event stream broke. Redis pub/sub keeps no
# history, so events published while a stream is down are gone; only the
# client can fill that gap, by reconnecting, resubscribing and refetching.
WS_CLOSE_EVENT_STREAM_LOST = 1011
WS_CLOSE_EVENT_STREAM_LOST_REASON = "Event stream interrupted"
# Handshakes are refused when a socket's event stream can't open, or too many
# are opening at once (e.g. everyone reconnecting after a Redis outage). Closing
# before accept() fails the handshake (HTTP 403), so the client never gets an
# open socket and keeps backing off; 1013 ("try again later") is the code the
# refusal is logged and counted under here.
WS_CLOSE_TRY_AGAIN_LATER = 1013
_MAX_CONCURRENT_STREAM_OPENS = 64


class _Subscription:
    """One SSUBSCRIBE lifecycle bound to a WebSocket, pinned to the owning shard."""

    def __init__(self, full_channel: str) -> None:
        _assert_no_wildcard(full_channel)
        self.full_channel = full_channel
        self._client: AsyncRedis | None = None
        self._pubsub: AsyncPubSub | None = None
        self._task: asyncio.Task | None = None
        self._lost = False
        self._stopping = False

    async def start(self, on_message: MessageHandler, on_lost: LostHandler) -> None:
        try:
            await asyncio.wait_for(self._open_pubsub(), _SUBSCRIBE_OPEN_TIMEOUT_S)
        except BaseException:
            await self._close_pubsub_quietly()
            raise
        self._task = asyncio.create_task(self._pump(on_message, on_lost))

    async def _open_pubsub(self) -> None:
        # Returns once Redis confirmed the SSUBSCRIBE, on a connection that
        # never reconnects silently: any drop reaches the pump as an error.
        self._client, self._pubsub = await redis.subscribe_sharded_async(
            self.full_channel
        )

    async def _close_pubsub_quietly(self) -> None:
        """Best-effort teardown — never raises."""
        if self._pubsub is not None:
            try:
                await self._pubsub.aclose()
            except Exception:
                pass
            self._pubsub = None
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def _pump(self, on_message: MessageHandler, on_lost: LostHandler) -> None:
        pubsub = self._pubsub
        if pubsub is None:
            return
        try:
            cause = await self._forward_until_lost(pubsub, on_message)
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError, RedisError) as exc:
            logger.warning(
                "Pubsub stream lost for %s: %s: %s",
                self.full_channel,
                type(exc).__name__,
                exc,
            )
            cause = "redis_error"
        except Exception:
            logger.exception("Pubsub pump crashed for %s", self.full_channel)
            cause = "crashed"
        self._lost = True
        if not self._stopping:
            await on_lost(cause)

    async def _forward_until_lost(
        self, pubsub: AsyncPubSub, on_message: MessageHandler
    ) -> str:
        async for message in pubsub.listen():
            msg_type = message.get("type")
            # Server-pushed sunsubscribe: the slot moved to another shard and
            # Redis revoked our SSUBSCRIBE without dropping the connection.
            if msg_type == "sunsubscribe":
                return "revoked"
            if msg_type not in ("smessage", "message", "pmessage"):
                continue
            try:
                await on_message(message.get("data"))
            except Exception:
                logger.exception(
                    "Websocket message-handler failed for channel %s",
                    self.full_channel,
                )
        return "ended"

    async def stop(self) -> None:
        # A deliberate stop never reports the stream as lost.
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # A lost stream's connection is already gone; SUNSUBSCRIBE would only
        # set off redis-py's reconnect retries.
        if self._pubsub is not None and not self._lost:
            try:
                await self._pubsub.sunsubscribe(self.full_channel)
            except Exception:
                logger.warning(
                    "SUNSUBSCRIBE failed for %s", self.full_channel, exc_info=True
                )
        await self._close_pubsub_quietly()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # channel_key → sockets subscribed (public channel keys, not raw Redis channels)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        # websocket → {channel_key: _Subscription}
        self._ws_subs: Dict[WebSocket, Dict[str, _Subscription]] = {}
        # websocket → notification subscription
        self._ws_notifications: Dict[WebSocket, _Subscription] = {}
        # sockets already being closed because an event stream was lost
        self._closing: Set[WebSocket] = set()
        # notification streams being opened for handshakes in progress
        self._opening = 0

    async def connect_socket(self, websocket: WebSocket, *, user_id: str) -> bool:
        """Accept the socket only once its notification stream is open, so a
        client never sees a connection that can't deliver events. Returns
        False when the handshake was refused."""
        if self._opening >= _MAX_CONCURRENT_STREAM_OPENS:
            await self._refuse(websocket, cause="busy", reason="Server busy")
            return False
        # Notifications that arrive before accept() wait for it instead of
        # failing on a socket that is still connecting.
        accepted = asyncio.Event()
        self._opening += 1
        try:
            opened = await self._start_notification_subscription(
                websocket, user_id=user_id, accepted=accepted
            )
        finally:
            self._opening -= 1
        if not opened or websocket in self._closing:
            await self._refuse(
                websocket, cause="open_failed", reason="Event stream unavailable"
            )
            return False
        await websocket.accept()
        accepted.set()
        self.active_connections.add(websocket)
        self._ws_subs.setdefault(websocket, {})
        return True

    async def _refuse(self, websocket: WebSocket, *, cause: str, reason: str) -> None:
        record_websocket_event_stream_lost(cause)
        logger.warning("Refusing WebSocket handshake: %s", reason)
        try:
            await websocket.close(code=WS_CLOSE_TRY_AGAIN_LATER, reason=reason)
        except Exception as e:
            if not _is_ws_close_race(e, websocket):
                logger.warning("Failed to refuse WebSocket", exc_info=True)

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
        # Closing for the whole teardown: a pump that loses its stream while
        # its siblings are being stopped must not report it (or re-add the
        # socket to _closing after we've cleared it).
        self._closing.add(websocket)
        try:
            # Stop SSUBSCRIBE pumps before dropping bookkeeping to avoid leaks.
            subs = self._ws_subs.pop(websocket, {})
            for sub in subs.values():
                await sub.stop()
            notif_sub = self._ws_notifications.pop(websocket, None)
            if notif_sub is not None:
                await notif_sub.stop()
            for channel_key, subscribers in list(self.subscriptions.items()):
                subscribers.discard(websocket)
                if not subscribers:
                    self.subscriptions.pop(channel_key, None)
        finally:
            self._closing.discard(websocket)

    async def subscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str:
        # Hash-tagged channel needs graph_id; resolve once per subscribe.
        meta = await get_graph_execution_meta(user_id, graph_exec_id)
        if meta is None:
            raise ValueError(
                f"graph_exec #{graph_exec_id} not found for user #{user_id}"
            )
        channel_key = graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        full_channel = event_bus_channel(
            exec_channel(user_id, meta.graph_id, graph_exec_id)
        )
        await self._open_subscription(websocket, channel_key, full_channel)
        return channel_key

    async def subscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        full_channel = event_bus_channel(graph_all_channel(user_id, graph_id))
        await self._open_subscription(websocket, channel_key, full_channel)
        return channel_key

    async def unsubscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        return await self._close_subscription(websocket, channel_key)

    async def unsubscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        return await self._close_subscription(websocket, channel_key)

    async def _open_subscription(
        self, websocket: WebSocket, channel_key: str, full_channel: str
    ) -> None:
        per_ws = self._ws_subs.setdefault(websocket, {})
        if channel_key in per_ws:
            return
        sub = _Subscription(full_channel)

        async def on_message(data: Optional[bytes | str]) -> None:
            await self._forward_exec_event(websocket, channel_key, data)

        async def on_lost(cause: str) -> None:
            await self.close_for_lost_stream(websocket, cause=cause)

        await sub.start(on_message, on_lost)
        per_ws[channel_key] = sub
        self.subscriptions.setdefault(channel_key, set()).add(websocket)

    async def close_for_lost_stream(self, websocket: WebSocket, *, cause: str) -> None:
        """Close a socket that can no longer receive events, so the client
        reconnects, resubscribes and refetches what it missed. Never keep a
        socket open (its heartbeats still answered) with a dead event stream."""
        if websocket in self._closing:
            return
        self._closing.add(websocket)
        record_websocket_event_stream_lost(cause)
        logger.warning(
            "Closing WebSocket after its event stream was lost (%s); "
            "the client will reconnect and resubscribe",
            cause,
        )
        try:
            await websocket.close(
                code=WS_CLOSE_EVENT_STREAM_LOST,
                reason=WS_CLOSE_EVENT_STREAM_LOST_REASON,
            )
        except Exception as e:
            if not _is_ws_close_race(e, websocket):
                logger.warning("Failed to close WebSocket", exc_info=True)

    async def _close_subscription(
        self, websocket: WebSocket, channel_key: str
    ) -> str | None:
        subscribers = self.subscriptions.get(channel_key)
        if subscribers is None:
            return None
        subscribers.discard(websocket)
        if not subscribers:
            self.subscriptions.pop(channel_key, None)
        per_ws = self._ws_subs.get(websocket)
        if per_ws and channel_key in per_ws:
            sub = per_ws.pop(channel_key)
            await sub.stop()
        return channel_key

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
        self,
        websocket: WebSocket,
        *,
        user_id: str,
        accepted: asyncio.Event | None = None,
    ) -> bool:
        full_channel = _notification_bus_channel(user_id)
        sub = _Subscription(full_channel)

        async def on_message(data: Optional[bytes | str]) -> None:
            if accepted is not None:
                await accepted.wait()
            await self._forward_notification(websocket, user_id, data)

        async def on_lost(cause: str) -> None:
            await self.close_for_lost_stream(websocket, cause=cause)

        try:
            await sub.start(on_message, on_lost)
        except Exception as e:
            logger.warning(
                "Failed to open notification SSUBSCRIBE for user=%s: %r", user_id, e
            )
            return False
        self._ws_notifications[websocket] = sub
        return True

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


def graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
