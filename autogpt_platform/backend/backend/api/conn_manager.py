import asyncio
import json
import logging
from typing import Awaitable, Callable, Dict, Optional, Set

from fastapi import WebSocket
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub

from backend.api.model import WSMessage, WSMethod
from backend.data import redis_client as redis
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import (
    ExecutionEventType,
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


MessageHandler = Callable[[Optional[bytes | str]], Awaitable[None]]


class _Subscription:
    """One SSUBSCRIBE lifecycle bound to a WebSocket, pinned to the owning shard."""

    def __init__(self, full_channel: str) -> None:
        _assert_no_wildcard(full_channel)
        self.full_channel = full_channel
        self._client: AsyncRedis | None = None
        self._pubsub: AsyncPubSub | None = None
        self._task: asyncio.Task | None = None

    async def start(self, on_message: MessageHandler) -> None:
        self._client = await redis.connect_sharded_pubsub_async(self.full_channel)
        self._pubsub = self._client.pubsub()
        await self._pubsub.execute_command("SSUBSCRIBE", self.full_channel)
        # redis-py 6.x async PubSub.listen() exits when ``channels`` is
        # empty; raw SSUBSCRIBE doesn't populate it, so do it ourselves.
        self._pubsub.channels[self.full_channel] = None  # type: ignore[index]
        self._task = asyncio.create_task(self._pump(on_message))

    async def _pump(self, on_message: MessageHandler) -> None:
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
        # websocket → {channel_key: _Subscription}
        self._ws_subs: Dict[WebSocket, Dict[str, _Subscription]] = {}
        # websocket → notification subscription
        self._ws_notifications: Dict[WebSocket, _Subscription] = {}

    async def connect_socket(self, websocket: WebSocket, *, user_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self._ws_subs.setdefault(websocket, {})
        await self._start_notification_subscription(websocket, user_id=user_id)

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
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

    async def subscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str:
        # Hash-tagged channel needs graph_id; resolve once per subscribe.
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

        async def on_message(data: Optional[bytes | str]) -> None:
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
        self,
        websocket: WebSocket,
        channel_key: str,
        raw_payload: Optional[bytes | str],
    ) -> None:
        if raw_payload is None:
            return
        # Payload is a serialized ``_EventPayloadWrapper[ExecutionEvent]``;
        # unwrap then re-wrap as a WS message.
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
        except Exception:
            logger.exception("Failed to forward exec event on %s", channel_key)

    async def _start_notification_subscription(
        self, websocket: WebSocket, *, user_id: str
    ) -> None:
        # One SSUBSCRIBE per WS; the pump delivers straight to its owning socket.
        full_channel = _notification_bus_channel(user_id)
        sub = _Subscription(full_channel)

        async def on_message(data: Optional[bytes | str]) -> None:
            await self._forward_notification(websocket, user_id, data)

        try:
            await sub.start(on_message)
        except Exception:
            logger.exception(
                "Failed to open notification SSUBSCRIBE for user=%s", user_id
            )
            return
        self._ws_notifications[websocket] = sub

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
            # Strip ``_EventPayloadWrapper`` envelope before validating.
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
        # Defense in depth: reject cross-user payloads.
        if event.user_id != user_id:
            return
        message = WSMessage(
            method=WSMethod.NOTIFICATION,
            data=event.payload.model_dump(),
        ).model_dump_json()
        try:
            await websocket.send_text(message)
        except Exception:
            logger.warning(
                "Failed to deliver notification to WS for user=%s",
                user_id,
                exc_info=True,
            )


def _graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
