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
from backend.data.db_accessors import execution_db
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import ExecutionEventType, exec_channel, graph_all_channel
from backend.data.notification_bus import NotificationEvent
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
# Opening a shard connection retries for minutes while Redis is down; give up
# sooner so the handler answers and the router reads the socket again.
_STREAM_OPEN_TIMEOUT_S = 15.0

WS_CLOSE_EVENT_STREAM_LOST = 1011
WS_CLOSE_TRY_AGAIN_LATER = 1013

Shard = tuple[str, int]


class _ShardStream:
    """One socket's SSUBSCRIBEs on one Redis shard, all over one connection."""

    def __init__(self, on_lost: LostHandler) -> None:
        # full channel → handler
        self.handlers: dict[str, MessageHandler] = {}
        self._on_lost = on_lost
        self._client: AsyncRedis | None = None
        self._pubsub: AsyncPubSub | None = None
        self._task: asyncio.Task | None = None
        # channels we SUNSUBSCRIBEd whose confirmation is still on its way
        self._unsubscribing: set[str] = set()
        self._stopping = False

    async def add(self, channel: str, handler: MessageHandler) -> None:
        _assert_no_wildcard(channel)
        pubsub = self._pubsub or await asyncio.wait_for(
            self._open(channel), _STREAM_OPEN_TIMEOUT_S
        )
        try:
            await pubsub.ssubscribe(channel)
        except BaseException:
            if not self.handlers:
                await self._close_quietly()
            raise
        self._unsubscribing.discard(channel)
        self.handlers[channel] = handler
        if self._task is None:
            self._task = asyncio.create_task(self._pump(pubsub))

    async def remove(self, channel: str) -> None:
        if self.handlers.pop(channel, None) is None:
            return
        if not self.handlers:
            await self.stop()
            return
        if self._pubsub is None:
            return
        self._unsubscribing.add(channel)
        try:
            await self._pubsub.sunsubscribe(channel)
        except Exception:
            logger.warning(f"SUNSUBSCRIBE failed for {channel}", exc_info=True)

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

    async def _open(self, channel: str) -> AsyncPubSub:
        self._client = await redis.connect_sharded_pubsub_async(channel)
        self._pubsub = self._client.pubsub()
        return self._pubsub

    async def _close_quietly(self) -> None:
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

    async def _pump(self, pubsub: AsyncPubSub) -> None:
        try:
            await self._forward(pubsub)
        except asyncio.CancelledError:
            raise
        except (OSError, RedisError) as e:
            logger.warning(f"Pubsub stream lost: {type(e).__name__}: {e}")
        except Exception:
            logger.exception("Pubsub pump crashed")
        if not self._stopping:
            await self._on_lost()

    async def _forward(self, pubsub: AsyncPubSub) -> None:
        async for message in pubsub.listen():
            msg_type = message.get("type")
            channel = message.get("channel")
            if msg_type == "sunsubscribe":
                if channel in self._unsubscribing:
                    self._unsubscribing.discard(channel)
                    continue
                # Not ours: the slot moved and Redis dropped the subscription.
                return
            if msg_type != "smessage":
                continue
            handler = self.handlers.get(channel) if channel else None
            if handler is None:
                continue
            try:
                await handler(message.get("data"))
            except Exception:
                logger.exception(f"Websocket message-handler failed for {channel}")


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
        # user_id → that user's sockets on this process
        self._user_sockets: dict[str, set[WebSocket]] = {}

    async def connect_socket(self, websocket: WebSocket, *, user_id: str) -> bool:
        """Accept the socket, or refuse the handshake and return False when the
        user already holds ``MAX_SOCKETS_PER_USER`` here."""
        sockets = self._user_sockets.setdefault(user_id, set())
        if len(sockets) >= MAX_SOCKETS_PER_USER:
            logger.warning(f"Refusing WebSocket for user #{user_id}: too many open")
            await websocket.close(
                code=WS_CLOSE_TRY_AGAIN_LATER, reason="Too many connections"
            )
            return False
        sockets.add(websocket)
        try:
            await websocket.accept()
        except BaseException:
            self._release_user_socket(user_id, websocket)
            raise
        self.active_connections.add(websocket)
        self._ws_subs.setdefault(websocket, {})
        await self._start_notification_subscription(websocket, user_id=user_id)
        return True

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
        self._release_user_socket(user_id, websocket)
        # Stop the shard streams before dropping bookkeeping to avoid leaks.
        for stream in self._ws_streams.pop(websocket, {}).values():
            await stream.stop()
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
        if channel_key not in per_ws:
            if len(per_ws) >= MAX_SUBSCRIPTIONS_PER_SOCKET:
                raise ValueError(
                    f"Subscription limit reached ({MAX_SUBSCRIPTIONS_PER_SOCKET} "
                    "per connection)"
                )

            async def on_message(data: Optional[bytes | str]) -> None:
                await self._forward_exec_event(websocket, channel_key, data)

            await self._subscribe(websocket, full_channel, on_message)
            per_ws[channel_key] = full_channel
        self.subscriptions.setdefault(channel_key, set()).add(websocket)

    async def _close_subscription(
        self, websocket: WebSocket, channel_key: str
    ) -> str | None:
        subscribers = self.subscriptions.get(channel_key)
        if subscribers is None:
            return None
        subscribers.discard(websocket)
        if not subscribers:
            self.subscriptions.pop(channel_key, None)
        full_channel = self._ws_subs.get(websocket, {}).pop(channel_key, None)
        if full_channel is not None:
            await self._unsubscribe(websocket, full_channel)
        return channel_key

    async def _subscribe(
        self, websocket: WebSocket, full_channel: str, handler: MessageHandler
    ) -> None:
        """SSUBSCRIBE on the socket's stream for the channel's shard, opening
        that stream if this is the socket's first channel there."""
        shard = redis.resolve_shard_for_channel(full_channel)
        streams = self._ws_streams.setdefault(websocket, {})
        stream = streams.get(shard)
        if stream is None:

            async def on_lost() -> None:
                await self._close_for_lost_stream(websocket)

            stream = streams[shard] = _ShardStream(on_lost)
        try:
            await stream.add(full_channel, handler)
        except BaseException:
            if not stream.handlers:
                streams.pop(shard, None)
            raise

    async def _unsubscribe(self, websocket: WebSocket, full_channel: str) -> None:
        streams = self._ws_streams.get(websocket, {})
        for shard, stream in list(streams.items()):
            if full_channel not in stream.handlers:
                continue
            await stream.remove(full_channel)
            if not stream.handlers:
                streams.pop(shard, None)
            return

    async def _close_for_lost_stream(self, websocket: WebSocket) -> None:
        """Close a socket whose events stopped arriving, so the client
        reconnects, resubscribes and refetches what it missed."""
        logger.warning("Closing WebSocket after its event stream was lost")
        try:
            await websocket.close(
                code=WS_CLOSE_EVENT_STREAM_LOST, reason="Event stream interrupted"
            )
        except Exception as e:
            if not _is_ws_close_race(e, websocket):
                logger.warning("Failed to close WebSocket", exc_info=True)

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

        try:
            await self._subscribe(websocket, full_channel, on_message)
        except Exception:
            logger.exception(f"Failed to open notification SSUBSCRIBE for {user_id}")
            return
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


def graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
