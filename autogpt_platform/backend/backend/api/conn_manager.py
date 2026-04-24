import asyncio
import json as _json
import logging
from typing import Dict, Set

from fastapi import WebSocket

from backend.api.model import NotificationPayload, WSMessage, WSMethod
from backend.data import redis_client as redis
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import (
    ExecutionEventType,
    GraphExecutionEvent,
    NodeExecutionEvent,
    _exec_channel,
    _graph_all_channel,
    get_graph_execution_meta,
)
from backend.data.notification_bus import NotificationEvent
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_settings = Settings()

_EVENT_TYPE_TO_METHOD_MAP: dict[ExecutionEventType, WSMethod] = {
    ExecutionEventType.GRAPH_EXEC_UPDATE: WSMethod.GRAPH_EXECUTION_EVENT,
    ExecutionEventType.NODE_EXEC_UPDATE: WSMethod.NODE_EXECUTION_EVENT,
}


def _event_bus_channel(channel_key: str) -> str:
    """Prefix a channel key with the execution event bus name."""
    return f"{_settings.config.execution_event_bus_name}/{channel_key}"


def _notification_bus_channel(user_id: str) -> str:
    """Return the full sharded channel name for a user's notifications."""
    return f"{_settings.config.notification_event_bus_name}/{user_id}"


class _Subscription:
    """One SSUBSCRIBE lifecycle bound to a WebSocket.

    Owns a shard-pinned ``Redis`` client because sharded pub/sub only
    delivers on the shard that owns the channel's keyslot. A subscribed
    connection cannot share a socket with regular commands, so each
    subscription gets its own connection.
    """

    def __init__(self, full_channel: str) -> None:
        _assert_no_wildcard(full_channel)
        self.full_channel = full_channel
        self._client = None
        self._pubsub = None
        self._task: asyncio.Task | None = None

    async def start(self, on_message) -> None:
        self._client = await redis.connect_sharded_pubsub_async(self.full_channel)
        self._pubsub = self._client.pubsub()
        await self._pubsub.execute_command("SSUBSCRIBE", self.full_channel)
        # redis-py 6.x async has no ssubscribe wrapper; pubsub.listen()
        # exits as soon as ``channels`` is empty, so populate it ourselves.
        self._pubsub.channels[self.full_channel] = None  # type: ignore[index]
        self._task = asyncio.create_task(self._pump(on_message))

    async def _pump(self, on_message) -> None:
        pubsub = self._pubsub
        if pubsub is None:
            return
        try:
            async for message in pubsub.listen():
                if message.get("type") not in ("smessage", "message", "pmessage"):
                    continue
                try:
                    await on_message(message.get("data"))
                except Exception:
                    logger.exception(
                        "Websocket message-handler failed for channel %s",
                        self.full_channel,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pubsub pump crashed for %s", self.full_channel)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._pubsub is not None:
            try:
                await self._pubsub.execute_command("SUNSUBSCRIBE", self.full_channel)
            except Exception:
                logger.warning(
                    "SUNSUBSCRIBE failed for %s", self.full_channel, exc_info=True
                )
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


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # channel_key → sockets subscribed (public channel keys, not raw Redis channels)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        # user_id → sockets (used by per-user notification fan-out)
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # websocket → {channel_key: _Subscription}
        self._ws_subs: Dict[WebSocket, Dict[str, _Subscription]] = {}
        # websocket → notification subscription
        self._ws_notifications: Dict[WebSocket, _Subscription] = {}

    async def connect_socket(self, websocket: WebSocket, *, user_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self._ws_subs.setdefault(websocket, {})
        self.user_connections.setdefault(user_id, set()).add(websocket)
        await self._start_notification_subscription(websocket, user_id=user_id)

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
        # Tear down every per-connection SSUBSCRIBE before dropping
        # bookkeeping so Redis doesn't leak sockets on client hangups.
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
        user_conns = self.user_connections.get(user_id)
        if user_conns is not None:
            user_conns.discard(websocket)
            if not user_conns:
                self.user_connections.pop(user_id, None)

    async def subscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str:
        # Sharded channels need the ``graph_id`` (part of the hash tag).
        # Resolve it via a single DB lookup the first time we subscribe.
        meta = await get_graph_execution_meta(user_id, graph_exec_id)
        if meta is None:
            raise ValueError(
                f"graph_exec #{graph_exec_id} not found for user #{user_id}"
            )
        channel_key = _graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        full_channel = _event_bus_channel(
            _exec_channel(user_id, meta.graph_id, graph_exec_id)
        )
        await self._open_subscription(websocket, channel_key, full_channel)
        return channel_key

    async def subscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        full_channel = _event_bus_channel(_graph_all_channel(user_id, graph_id))
        await self._open_subscription(websocket, channel_key, full_channel)
        return channel_key

    async def unsubscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = _graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        return await self._close_subscription(websocket, channel_key)

    async def unsubscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = _graph_execs_channel_key(user_id, graph_id=graph_id)
        return await self._close_subscription(websocket, channel_key)

    async def _open_subscription(
        self, websocket: WebSocket, channel_key: str, full_channel: str
    ) -> None:
        self.subscriptions.setdefault(channel_key, set()).add(websocket)
        per_ws = self._ws_subs.setdefault(websocket, {})
        if channel_key in per_ws:
            return
        sub = _Subscription(full_channel)

        async def on_message(data):
            await self._forward_exec_event(websocket, channel_key, data)

        await sub.start(on_message)
        per_ws[channel_key] = sub

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
        self, websocket: WebSocket, channel_key: str, raw_payload
    ) -> None:
        if raw_payload is None:
            return
        # ``raw_payload`` is the serialized ``_EventPayloadWrapper[ExecutionEvent]``
        # emitted by AsyncRedisExecutionEventBus. Unwrap it and re-wrap as a
        # WS message so the client sees the same shape as before.
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
            parsed = _json.loads(wrapper)
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
        except Exception:
            logger.exception("Failed to forward exec event on %s", channel_key)

    async def _start_notification_subscription(
        self, websocket: WebSocket, *, user_id: str
    ) -> None:
        # Each connected user gets a single SSUBSCRIBE on their notification
        # channel; fan-out to multiple tabs/devices happens in-process via
        # ``user_connections`` because notifications are user-scoped rather
        # than connection-scoped.
        full_channel = _notification_bus_channel(user_id)
        sub = _Subscription(full_channel)

        async def on_message(data):
            await self._forward_notification(user_id, data)

        try:
            await sub.start(on_message)
        except Exception:
            logger.exception(
                "Failed to open notification SSUBSCRIBE for user=%s", user_id
            )
            return
        self._ws_notifications[websocket] = sub

    async def _forward_notification(self, user_id: str, raw_payload) -> None:
        if raw_payload is None:
            return
        try:
            wrapper_json = (
                raw_payload.decode()
                if isinstance(raw_payload, (bytes, bytearray))
                else raw_payload
            )
            # Messages on the wire are wrapped by ``_EventPayloadWrapper``
            # ({"payload": {...NotificationEvent...}}); strip one level
            # before validating.
            parsed = _json.loads(wrapper_json)
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
        # Publisher already scopes to ``user_id``; skip any cross-user
        # leakage just in case.
        if event.user_id != user_id:
            return
        await self.send_notification(user_id=user_id, payload=event.payload)

    async def send_execution_update(
        self, exec_event: GraphExecutionEvent | NodeExecutionEvent
    ) -> int:
        """Route an in-process event (tests/back-compat) to local subscribers.

        Production delivery goes through the per-connection SSUBSCRIBE path
        in ``_open_subscription``; this helper lets unit tests exercise the
        app-level fan-out without a live Redis cluster.
        """
        graph_exec_id = (
            exec_event.id
            if isinstance(exec_event, GraphExecutionEvent)
            else exec_event.graph_exec_id
        )

        n_sent = 0

        channels: set[str] = {
            _graph_exec_channel_key(exec_event.user_id, graph_exec_id=graph_exec_id)
        }
        if isinstance(exec_event, GraphExecutionEvent):
            channels.add(
                _graph_execs_channel_key(
                    exec_event.user_id, graph_id=exec_event.graph_id
                )
            )

        for channel in channels.intersection(self.subscriptions.keys()):
            message = WSMessage(
                method=_EVENT_TYPE_TO_METHOD_MAP[exec_event.event_type],
                channel=channel,
                data=exec_event.model_dump(),
            ).model_dump_json()
            for connection in self.subscriptions[channel]:
                await connection.send_text(message)
                n_sent += 1

        return n_sent

    async def send_notification(
        self, *, user_id: str, payload: NotificationPayload
    ) -> int:
        """Send a notification to all websocket connections belonging to a user."""
        message = WSMessage(
            method=WSMethod.NOTIFICATION,
            data=payload.model_dump(),
        ).model_dump_json()

        connections = tuple(self.user_connections.get(user_id, set()))
        if not connections:
            return 0

        await asyncio.gather(
            *(connection.send_text(message) for connection in connections),
            return_exceptions=True,
        )

        return len(connections)


def _graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
