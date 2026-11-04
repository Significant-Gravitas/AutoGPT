import asyncio
import json
import logging
from collections import Counter
from typing import Awaitable, Callable, Dict, Optional, Set

from autogpt_libs.auth.permissions import TeamAction
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.exceptions import RedisError
from starlette.websockets import WebSocketState

from backend.api.model import WSMessage, WSMethod
from backend.api.org_rollout import require_personal_scope_without_collaboration
from backend.data import db
from backend.data import redis_client as redis
from backend.data import tenancy
from backend.data.db_accessors import execution_db, orgs_db
from backend.data.event_bus import _assert_no_wildcard
from backend.data.execution import (
    ExecutionEventType,
    GraphExecutionMeta,
    exec_channel,
    graph_all_channel,
)
from backend.data.notification_bus import NotificationEvent
from backend.util.clients import get_database_manager_async_client
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
# Opening a shard connection retries for minutes while Redis is down; give up
# sooner so the handler answers and the router reads the socket again.
_STREAM_OPEN_TIMEOUT_S = 15.0

WS_CLOSE_EVENT_STREAM_LOST = 1011
WS_CLOSE_TRY_AGAIN_LATER = 1013

Shard = tuple[str, int]


class _ExecutionSubscriptionScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    organization_id: str | None
    team_id: str | None
    graph_id: str
    execution_id: str | None
    # VIEW_EXECUTIONS or VIEW_AGENTS; both are the "view" resource access.
    team_action: TeamAction


# The websocket server connects no Prisma client (it is given no database URL),
# so every read below goes through the DatabaseManager, as `execution_db()` does.


def _tenancy_db():
    if db.is_connected():
        return tenancy
    return get_database_manager_async_client()


async def get_graph_execution_exact_scope(
    user_id: str,
    execution_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> GraphExecutionMeta | None:
    """The user's execution, only if it is in exactly this org and team."""
    execution = await execution_db().get_graph_execution_meta(
        user_id=user_id, execution_id=execution_id
    )
    if (
        execution is None
        or execution.organization_id != organization_id
        or execution.team_id != team_id
    ):
        return None
    return execution


async def _user_default_team(user_id: str) -> tuple[str | None, str | None]:
    return await orgs_db().get_user_default_team(user_id)


class _StreamCapacityError(ValueError):
    """A shard had no room left for this user's stream."""


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # channel_key → sockets subscribed (public channel keys, not raw Redis channels)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        # websocket → {channel_key: full Redis channel}
        self._ws_subs: dict[WebSocket, dict[str, str]] = {}
        # websocket → {full Redis channel: {channel_key: the scope it streams}};
        # one graph's channel carries every org/team scope it was subscribed in
        self._ws_scopes: dict[
            WebSocket, dict[str, dict[str, _ExecutionSubscriptionScope]]
        ] = {}
        # websocket → its notification channel
        self._ws_notifications: dict[WebSocket, str] = {}
        # websocket → {shard: the one connection it subscribes over there}
        self._ws_streams: dict[WebSocket, dict[Shard, _ShardStream]] = {}
        # user_id → that user's sockets on this process
        self._user_sockets: dict[str, set[WebSocket]] = {}
        # shard → shard streams open to it from this process
        self._streams_per_shard: Counter[Shard] = Counter()
        # (user_id, shard) → that user's streams open to it
        self._user_streams: Counter[tuple[str, Shard]] = Counter()

    async def connect_socket(self, websocket: WebSocket, *, user_id: str) -> bool:
        """Accept the socket, or close it with 1013 and return False when the
        user holds ``MAX_SOCKETS_PER_USER`` here or a shard has no room left
        for the user's streams."""
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
        try:
            await self._start_notification_subscription(websocket, user_id=user_id)
        except _StreamCapacityError:
            # Capacity moved between admission and here; don't hand back a
            # socket that will never receive a notification.
            try:
                await websocket.close(
                    code=WS_CLOSE_TRY_AGAIN_LATER, reason="Try again later"
                )
            finally:
                await self.disconnect_socket(websocket, user_id=user_id)
            return False
        return True

    async def disconnect_socket(self, websocket: WebSocket, *, user_id: str):
        self.active_connections.discard(websocket)
        self._release_user_socket(user_id, websocket)
        # Stop the shard streams before dropping bookkeeping to avoid leaks.
        for shard, stream in self._ws_streams.pop(websocket, {}).items():
            await stream.stop()
            self._release_stream(user_id, shard)
        self._ws_notifications.pop(websocket, None)
        self._ws_scopes.pop(websocket, None)
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
        scope = _ExecutionSubscriptionScope(
            user_id=user_id,
            organization_id=meta.organization_id,
            team_id=meta.team_id,
            graph_id=meta.graph_id,
            execution_id=graph_exec_id,
            team_action=TeamAction.VIEW_EXECUTIONS,
        )
        if not await self._execution_scope_is_live(scope):
            raise ValueError("Access denied")
        if scope.organization_id is not None or scope.team_id is not None:
            await require_personal_scope_without_collaboration(
                user_id,
                scope.organization_id,
                scope.team_id,
                default_team=_user_default_team,
            )
        await self._open_subscription(
            user_id, websocket, channel_key, full_channel, scope
        )
        return channel_key

    async def subscribe_graph_execs(
        self,
        *,
        user_id: str,
        graph_id: str,
        organization_id: str | None,
        team_id: str | None,
        websocket: WebSocket,
    ) -> str:
        scope = _ExecutionSubscriptionScope(
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
            graph_id=graph_id,
            execution_id=None,
            team_action=TeamAction.VIEW_AGENTS,
        )
        channel_key = _graph_execs_channel_key(
            user_id,
            graph_id=graph_id,
            organization_id=organization_id,
            team_id=team_id,
        )
        full_channel = event_bus_channel(graph_all_channel(user_id, graph_id))
        if not await self._execution_scope_is_live(scope):
            raise ValueError("Access denied")
        # Fully unscoped subscriptions are legacy personal channels; the
        # exact-scope filter still excludes every org/team execution from them.
        if organization_id is not None or team_id is not None:
            await require_personal_scope_without_collaboration(
                user_id, organization_id, team_id, default_team=_user_default_team
            )
        await self._open_subscription(
            user_id, websocket, channel_key, full_channel, scope
        )
        return channel_key

    async def unsubscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        channel_key = graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id)
        return await self._close_subscription(user_id, websocket, channel_key)

    async def unsubscribe_graph_execs(
        self,
        *,
        user_id: str,
        graph_id: str,
        organization_id: str | None,
        team_id: str | None,
        websocket: WebSocket,
    ) -> str | None:
        channel_key = _graph_execs_channel_key(
            user_id,
            graph_id=graph_id,
            organization_id=organization_id,
            team_id=team_id,
        )
        return await self._close_subscription(user_id, websocket, channel_key)

    async def _open_subscription(
        self,
        user_id: str,
        websocket: WebSocket,
        channel_key: str,
        full_channel: str,
        scope: _ExecutionSubscriptionScope,
    ) -> None:
        per_ws = self._ws_subs.setdefault(websocket, {})
        if channel_key not in per_ws:
            if len(per_ws) >= MAX_SUBSCRIPTIONS_PER_SOCKET:
                raise ValueError(
                    f"Subscription limit reached ({MAX_SUBSCRIPTIONS_PER_SOCKET} "
                    "per connection)"
                )
            # A graph's channel already followed in another org/team scope
            # keeps its one SSUBSCRIBE; its handler serves every scope.
            if not self._ws_scopes.get(websocket, {}).get(full_channel):

                async def on_message(data: Optional[bytes | str]) -> None:
                    await self._forward_scoped_exec_event(websocket, full_channel, data)

                await self._subscribe(user_id, websocket, full_channel, on_message)
            scopes = self._ws_scopes.setdefault(websocket, {})
            scopes.setdefault(full_channel, {})[channel_key] = scope
            per_ws[channel_key] = full_channel
        self.subscriptions.setdefault(channel_key, set()).add(websocket)

    async def _forward_scoped_exec_event(
        self,
        websocket: WebSocket,
        full_channel: str,
        data: Optional[bytes | str],
    ) -> None:
        scopes = self._ws_scopes.get(websocket, {}).get(full_channel, {})
        for channel_key, scope in list(scopes.items()):
            # Checked on every event, so a revoked membership or role ends the
            # stream at its next event.
            if not await self._execution_scope_is_live(scope):
                await self._close_revoked_execution_websocket(websocket)
                return
            await self._forward_exec_event(
                websocket,
                channel_key,
                data,
                required_scope=scope,
            )

    async def _close_revoked_execution_websocket(self, websocket: WebSocket) -> None:
        try:
            await websocket.close(
                code=4003,
                reason="Execution subscription access was revoked",
            )
        except Exception as error:
            if not _is_ws_close_race(error, websocket):
                logger.warning(
                    "Failed to close revoked execution subscription",
                    exc_info=True,
                )

    async def _execution_scope_is_live(
        self,
        scope: _ExecutionSubscriptionScope,
    ) -> bool:
        """Whether the subscriber may still see what this scope streams: a live
        membership and role in its org and team, and, for one execution, that
        execution still in exactly that scope."""
        if (
            scope.organization_id is not None
            and not await _tenancy_db().has_live_resource_access(
                scope.user_id, scope.organization_id, scope.team_id, "view"
            )
        ):
            return False
        if scope.execution_id is None:
            return True
        execution = await get_graph_execution_exact_scope(
            scope.user_id,
            scope.execution_id,
            scope.organization_id,
            scope.team_id,
        )
        return execution is not None and execution.graph_id == scope.graph_id

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
        if full_channel is None:
            return channel_key
        scopes = self._ws_scopes.get(websocket, {})
        channel_scopes = scopes.get(full_channel, {})
        channel_scopes.pop(channel_key, None)
        if not channel_scopes:
            scopes.pop(full_channel, None)
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
        shard = redis.resolve_shard_for_channel(full_channel)
        streams = self._ws_streams.setdefault(websocket, {})
        stream = streams.get(shard)
        if stream is None:
            if not self._has_room(user_id, shard):
                raise _StreamCapacityError("Server busy, try again later")

            async def on_lost() -> None:
                await self._close_for_lost_stream(websocket)

            stream = streams[shard] = _ShardStream(on_lost)
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
        required_scope: _ExecutionSubscriptionScope | None = None,
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
            if required_scope is not None:
                execution_id = (
                    event_data.get("id")
                    if method == WSMethod.GRAPH_EXECUTION_EVENT
                    else event_data.get("graph_exec_id")
                )
                if not isinstance(execution_id, str):
                    return
                execution = await get_graph_execution_exact_scope(
                    required_scope.user_id,
                    execution_id,
                    required_scope.organization_id,
                    required_scope.team_id,
                )
                if (
                    execution is None
                    or execution.graph_id != required_scope.graph_id
                    or (
                        required_scope.execution_id is not None
                        and execution.id != required_scope.execution_id
                    )
                ):
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
            await self._subscribe(user_id, websocket, full_channel, on_message)
        except _StreamCapacityError:
            raise
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


class _ShardStream:
    """One socket's SSUBSCRIBEs on one Redis shard, all over one connection."""

    def __init__(self, on_lost: LostHandler) -> None:
        # full channel → handler
        self.handlers: dict[str, MessageHandler] = {}
        self._on_lost = on_lost
        self._client: AsyncRedis | None = None
        self._pubsub: AsyncPubSub | None = None
        self._task: asyncio.Task | None = None
        # SUNSUBSCRIBEs sent per channel whose confirmation is still on its way
        self._unsubscribing: Counter[str] = Counter()
        self._stopping = False

    async def add(self, channel: str, handler: MessageHandler) -> None:
        _assert_no_wildcard(channel)
        try:
            pubsub = await asyncio.wait_for(
                self._open_and_subscribe(channel), _STREAM_OPEN_TIMEOUT_S
            )
        except BaseException:
            if not self.handlers:
                await self._close_quietly()
            raise
        self.handlers[channel] = handler
        if self._task is None:
            self._task = asyncio.create_task(self._pump(pubsub))

    async def _open_and_subscribe(self, channel: str) -> AsyncPubSub:
        """``asyncio.wait_for`` bounds this whole call, not just opening the
        connection — SSUBSCRIBE has no socket timeout of its own."""
        pubsub = self._pubsub or await self._open(channel)
        await pubsub.ssubscribe(channel)
        return pubsub

    async def remove(self, channel: str) -> None:
        if self.handlers.pop(channel, None) is None:
            return
        if not self.handlers:
            await self.stop()
            return
        if self._pubsub is None:
            return
        self._unsubscribing[channel] += 1
        try:
            await self._pubsub.sunsubscribe(channel)
        except Exception:
            # No confirmation is coming; a stale count would hide a real drop.
            self._unsubscribing[channel] -= 1
            if self._unsubscribing[channel] <= 0:
                del self._unsubscribing[channel]
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
                if self._unsubscribing[channel] > 0:
                    self._unsubscribing[channel] -= 1
                    if self._unsubscribing[channel] <= 0:
                        del self._unsubscribing[channel]
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


def graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(
    user_id: str,
    *,
    graph_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> str:
    organization_key = organization_id or "__personal__"
    team_key = team_id or "__org_home__"
    return (
        f"{user_id}|org#{organization_key}|team#{team_key}|graph#{graph_id}|executions"
    )
