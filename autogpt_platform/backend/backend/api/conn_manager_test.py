import ast
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.api.conn_manager import (
    MAX_SOCKETS_PER_USER,
    MAX_SUBSCRIPTIONS_PER_SOCKET,
    WS_CLOSE_EVENT_STREAM_LOST,
    WS_CLOSE_TRY_AGAIN_LATER,
    ConnectionManager,
    _graph_execs_channel_key,
    _notification_bus_channel,
    _ShardStream,
    event_bus_channel,
    graph_exec_channel_key,
)
from backend.api.model import WSMethod
from backend.data.execution import GraphExecutionMeta


class _FakePubSub:
    def __init__(self) -> None:
        self.ssubscribe = AsyncMock()
        self.sunsubscribe = AsyncMock()
        self.aclose = AsyncMock()
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()

    async def listen(self):
        while True:
            item = await self.inbox.get()
            if isinstance(item, BaseException):
                raise item
            yield item


class _FakeRedis:
    """Stands in for the shard-pinned connections; records each one opened."""

    def __init__(self) -> None:
        self.shard_of: Callable[[str], tuple[str, int]] = lambda _: ("redis", 1)
        self.clients: list[MagicMock] = []
        self.pubsubs: list[_FakePubSub] = []

    async def connect(self, channel: str) -> MagicMock:
        pubsub = _FakePubSub()
        client = MagicMock(pubsub=MagicMock(return_value=pubsub), aclose=AsyncMock())
        self.clients.append(client)
        self.pubsubs.append(pubsub)
        return client

    def subscribed(self) -> list[str]:
        return [
            call.args[0] for p in self.pubsubs for call in p.ssubscribe.await_args_list
        ]


@pytest.fixture
def fake_redis():
    fake = _FakeRedis()
    with (
        patch(
            "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
            side_effect=fake.connect,
        ),
        patch(
            "backend.api.conn_manager.redis.resolve_shard_for_channel",
            side_effect=lambda channel: fake.shard_of(channel),
        ),
    ):
        yield fake


@pytest.fixture
async def connection_manager():
    cm = ConnectionManager()
    yield cm
    # Stop every pump a test left running.
    for ws in list(cm._ws_streams):
        await cm.disconnect_socket(ws, user_id="user-1")


@pytest.fixture
def mock_websocket() -> AsyncMock:
    websocket: AsyncMock = AsyncMock(spec=WebSocket)
    websocket.send_text = AsyncMock()
    return websocket


def _meta(graph_id: str = "test_graph") -> MagicMock:
    meta = MagicMock(spec=GraphExecutionMeta)
    meta.id = "graph-exec-1"
    meta.user_id = "user-1"
    meta.graph_id = graph_id
    return meta


@pytest.mark.asyncio
async def test_connect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    # Stub the SSUBSCRIBE the test has no Redis for.
    with patch.object(
        connection_manager, "_start_notification_subscription", AsyncMock()
    ):
        accepted = await connection_manager.connect_socket(
            mock_websocket, user_id="user-1"
        )
    assert accepted is True
    assert mock_websocket in connection_manager.active_connections
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.active_connections.add(mock_websocket)
    connection_manager._ws_subs[mock_websocket] = {"test_channel_42": "full"}
    connection_manager.subscriptions["test_channel_42"] = {mock_websocket}

    await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")

    assert mock_websocket not in connection_manager.active_connections
    assert "test_channel_42" not in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_subscribe(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    with patch(
        "backend.api.conn_manager.execution_db",
        return_value=_execution_db(_meta()),
    ):
        await connection_manager.subscribe_graph_exec(
            user_id="user-1",
            graph_exec_id="graph-exec-1",
            websocket=mock_websocket,
        )
    assert (
        mock_websocket
        in connection_manager.subscriptions["user-1|graph_exec#graph-exec-1"]
    )
    assert fake_redis.subscribed() == [
        event_bus_channel("{user-1/test_graph}/exec/graph-exec-1")
    ]


@pytest.mark.asyncio
async def test_unsubscribe(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-1", websocket=mock_websocket
    )
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-2", websocket=mock_websocket
    )

    await connection_manager.unsubscribe_graph_execs(
        user_id="user-1", graph_id="graph-1", websocket=mock_websocket
    )

    assert "user-1|graph#graph-1|executions" not in connection_manager.subscriptions
    assert "user-1|graph#graph-2|executions" in connection_manager.subscriptions
    fake_redis.pubsubs[0].sunsubscribe.assert_awaited_once_with(
        event_bus_channel("{user-1/graph-1}/all")
    )


# ---------- Channel-name helpers ----------


def test_event_bus_channel_prefixes_settings_name() -> None:
    """Every SSUBSCRIBE must go through the bus-namespaced channel."""
    full = event_bus_channel("{u/g}/exec/x")
    # Must start with the configured bus name, then slash, then key.
    assert full.endswith("/{u/g}/exec/x")
    assert "/" in full


def test_notification_bus_channel_prefixes_settings_name() -> None:
    """Notification channels are ``<bus>/<user_id>`` — no wildcard."""
    full = _notification_bus_channel("user-42")
    assert full.endswith("/user-42")
    assert "*" not in full


def test_graph_exec_channel_key_shape() -> None:
    assert graph_exec_channel_key("u", graph_exec_id="e") == "u|graph_exec#e"


def test_graph_execs_channel_key_shape() -> None:
    assert _graph_execs_channel_key("u", graph_id="g") == "u|graph#g|executions"


# ---------- One Redis connection per shard per socket ----------


@pytest.mark.asyncio
async def test_a_sockets_subscriptions_share_one_connection_per_shard(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """A socket's subscriptions share one Redis connection per shard."""
    shards = [("redis", 17000), ("redis", 17001), ("redis", 17002)]
    fake_redis.shard_of = lambda channel: shards[sum(map(ord, channel)) % 3]

    with patch.object(connection_manager, "_forward_notification", AsyncMock()):
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    for i in range(60):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=f"graph-{i}", websocket=mock_websocket
        )

    assert len(fake_redis.subscribed()) == 61  # 60 graphs + notifications
    assert len(fake_redis.clients) == 3


@pytest.mark.asyncio
async def test_two_connections_get_independent_streams(
    connection_manager: ConnectionManager, fake_redis: _FakeRedis
) -> None:
    """Each socket subscribes over its own connection, so one socket's slow
    reads or lost stream never stall another's events."""
    ws_a: AsyncMock = AsyncMock(spec=WebSocket)
    ws_b: AsyncMock = AsyncMock(spec=WebSocket)
    with patch(
        "backend.api.conn_manager.execution_db",
        return_value=_execution_db(_meta()),
    ):
        await connection_manager.subscribe_graph_exec(
            user_id="user-1", graph_exec_id="graph-exec-1", websocket=ws_a
        )
        await connection_manager.subscribe_graph_exec(
            user_id="user-1", graph_exec_id="graph-exec-1", websocket=ws_b
        )

    assert len(fake_redis.clients) == 2
    subs = connection_manager.subscriptions["user-1|graph_exec#graph-exec-1"]
    assert ws_a in subs and ws_b in subs


def test_only_the_shard_stream_opens_pubsub_connections() -> None:
    """Only ``_ShardStream`` opens pub/sub connections, so they stay one per
    shard per socket."""
    api_dir = Path(__file__).parent
    openers = {"connect_sharded_pubsub_async", "connect_sharded_pubsub"}
    callers: set[str] = set()
    for path in api_dir.rglob("*.py"):
        if path.name.endswith("_test.py"):
            continue
        callers |= _callers_of(ast.parse(path.read_text()), openers, path.stem)
    assert callers == {"conn_manager._ShardStream._open"}


def _callers_of(tree: ast.Module, names: set[str], module: str) -> set[str]:
    found: set[str] = set()

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = [*scope, child.name]
            if isinstance(child, ast.Call):
                func = child.func
                name = func.attr if isinstance(func, ast.Attribute) else None
                if isinstance(func, ast.Name):
                    name = func.id
                if name in names:
                    found.add(".".join([module, *scope]))
            visit(child, inner)

    visit(tree, [])
    return found


# ---------- Per-socket and per-user limits ----------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["graph_execs", "graph_exec"])
async def test_subscriptions_per_socket_are_capped(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    method: str,
) -> None:
    async def subscribe(i: int) -> str:
        if method == "graph_execs":
            return await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id=f"g-{i}", websocket=mock_websocket
            )
        return await connection_manager.subscribe_graph_exec(
            user_id="user-1", graph_exec_id=f"e-{i}", websocket=mock_websocket
        )

    with patch(
        "backend.api.conn_manager.execution_db",
        return_value=_execution_db(_meta()),
    ):
        for i in range(MAX_SUBSCRIPTIONS_PER_SOCKET):
            await subscribe(i)
        with pytest.raises(ValueError, match="Subscription limit"):
            await subscribe(MAX_SUBSCRIPTIONS_PER_SOCKET)
        # Repeating a subscription the socket already holds is still answered.
        await subscribe(0)

    assert len(fake_redis.subscribed()) == MAX_SUBSCRIPTIONS_PER_SOCKET
    assert len(connection_manager._ws_subs[mock_websocket]) == (
        MAX_SUBSCRIPTIONS_PER_SOCKET
    )


@pytest.mark.asyncio
async def test_sockets_per_user_are_capped(
    connection_manager: ConnectionManager,
) -> None:
    sockets = [AsyncMock(spec=WebSocket) for _ in range(MAX_SOCKETS_PER_USER + 1)]
    with patch.object(
        connection_manager, "_start_notification_subscription", AsyncMock()
    ):
        for ws in sockets[:-1]:
            assert await connection_manager.connect_socket(ws, user_id="user-1")
        refused = sockets[-1]
        assert not await connection_manager.connect_socket(refused, user_id="user-1")
        refused.accept.assert_not_called()
        refused.close.assert_awaited_once()
        assert refused.close.await_args.kwargs["code"] == WS_CLOSE_TRY_AGAIN_LATER

        # Another user is unaffected, and a closed socket frees its slot.
        assert await connection_manager.connect_socket(
            AsyncMock(spec=WebSocket), user_id="user-2"
        )
        await connection_manager.disconnect_socket(sockets[0], user_id="user-1")
        assert await connection_manager.connect_socket(refused, user_id="user-1")


@pytest.mark.asyncio
async def test_a_refused_accept_frees_the_users_slot(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    mock_websocket.accept.side_effect = RuntimeError("client went away")
    with pytest.raises(RuntimeError):
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    assert "user-1" not in connection_manager._user_sockets


# ---------- _ShardStream ----------


@pytest.mark.asyncio
async def test_stream_subscribes_each_channel_on_one_connection(
    fake_redis: _FakeRedis,
) -> None:
    stream = _ShardStream(on_lost=AsyncMock())
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/g}/exec/b", AsyncMock())
        assert len(fake_redis.clients) == 1
        assert fake_redis.subscribed() == ["{u/g}/exec/a", "{u/g}/exec/b"]
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_rejects_wildcard_channel_before_redis(
    fake_redis: _FakeRedis,
) -> None:
    with pytest.raises(ValueError):
        await _ShardStream(on_lost=AsyncMock()).add("{u/g}/exec/*", AsyncMock())
    assert fake_redis.clients == []


@pytest.mark.asyncio
async def test_stream_forwards_smessages_to_their_channels_handler(
    fake_redis: _FakeRedis,
) -> None:
    on_a, on_b = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock())
    try:
        await stream.add("{u/g}/exec/a", on_a)
        await stream.add("{u/g}/exec/b", on_b)
        inbox = fake_redis.pubsubs[0].inbox
        for message in (
            {"type": "ssubscribe", "channel": "{u/g}/exec/a", "data": 1},
            {"type": "smessage", "channel": "{u/g}/exec/elsewhere", "data": "x"},
            {"type": "smessage", "channel": "{u/g}/exec/b", "data": "for-b"},
            {"type": "smessage", "channel": "{u/g}/exec/a", "data": "for-a"},
        ):
            inbox.put_nowait(message)
        await _until(lambda: on_a.await_count == 1)
        on_a.assert_awaited_once_with("for-a")
        on_b.assert_awaited_once_with("for-b")
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_survives_a_failing_handler(fake_redis: _FakeRedis) -> None:
    handler = AsyncMock(side_effect=[RuntimeError("boom"), None])
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost)
    try:
        await stream.add("{u/g}/exec/a", handler)
        inbox = fake_redis.pubsubs[0].inbox
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/a", "data": 1})
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/a", "data": 2})
        await _until(lambda: handler.await_count == 2)
        on_lost.assert_not_awaited()
    finally:
        await stream.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        RedisConnectionError("shard gone"),
        RuntimeError("transport closed"),
        {"type": "sunsubscribe", "channel": "{u/g}/exec/a", "data": 0},
    ],
    ids=["redis-error", "crash", "server-sunsubscribe"],
)
async def test_stream_reports_a_lost_stream(fake_redis: _FakeRedis, event: Any) -> None:
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        fake_redis.pubsubs[0].inbox.put_nowait(event)
        await _until(lambda: on_lost.await_count == 1)
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_keeps_flowing_after_its_own_sunsubscribe(
    fake_redis: _FakeRedis,
) -> None:
    on_b = AsyncMock()
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/g}/exec/b", on_b)
        await stream.remove("{u/g}/exec/a")
        inbox = fake_redis.pubsubs[0].inbox
        inbox.put_nowait({"type": "sunsubscribe", "channel": "{u/g}/exec/a"})
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/b", "data": 1})
        await _until(lambda: on_b.await_count == 1)
        on_lost.assert_not_awaited()
        fake_redis.pubsubs[0].sunsubscribe.assert_awaited_once_with("{u/g}/exec/a")
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_removing_the_last_channel_closes_the_connection(
    fake_redis: _FakeRedis,
) -> None:
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost)
    await stream.add("{u/g}/exec/a", AsyncMock())

    await stream.remove("{u/g}/exec/a")

    fake_redis.pubsubs[0].aclose.assert_awaited_once()
    fake_redis.clients[0].aclose.assert_awaited_once()
    on_lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_first_subscribe_closes_the_connection(
    fake_redis: _FakeRedis,
) -> None:
    stream = _ShardStream(on_lost=AsyncMock())
    original_connect = fake_redis.connect

    async def connect_then_fail(channel: str) -> MagicMock:
        client = await original_connect(channel)
        fake_redis.pubsubs[-1].ssubscribe.side_effect = RedisConnectionError("full")
        return client

    with (
        patch(
            "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
            side_effect=connect_then_fail,
        ),
        pytest.raises(RedisConnectionError),
    ):
        await stream.add("{u/g}/exec/a", AsyncMock())

    assert stream.handlers == {}
    fake_redis.clients[0].aclose.assert_awaited_once()


# ---------- ConnectionManager subscribe reuse / missing-meta handling ----------


@pytest.mark.asyncio
async def test_subscribe_graph_exec_raises_when_meta_missing(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """If graph_id can't be resolved, subscribe must reject — no silent SSUBSCRIBE."""
    with patch(
        "backend.api.conn_manager.execution_db",
        return_value=_execution_db(None),
    ):
        with pytest.raises(ValueError):
            await connection_manager.subscribe_graph_exec(
                user_id="user-1",
                graph_exec_id="unknown-exec",
                websocket=mock_websocket,
            )
    assert fake_redis.subscribed() == []


@pytest.mark.asyncio
async def test_subscribe_graph_exec_reuses_existing_subscription(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """A second subscribe on the same (ws, channel) must not SSUBSCRIBE again."""
    with patch(
        "backend.api.conn_manager.execution_db",
        return_value=_execution_db(_meta()),
    ):
        for _ in range(2):
            await connection_manager.subscribe_graph_exec(
                user_id="user-1",
                graph_exec_id="graph-exec-1",
                websocket=mock_websocket,
            )

    assert len(fake_redis.subscribed()) == 1


@pytest.mark.asyncio
async def test_subscribe_graph_execs_opens_aggregate_channel(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """subscribe_graph_execs must SSUBSCRIBE on the aggregate ``/all`` channel."""
    await connection_manager.subscribe_graph_execs(
        user_id="user-1",
        graph_id="graph-1",
        websocket=mock_websocket,
    )

    # Aggregate channel uses {user_id/graph_id} hash tag + /all suffix.
    assert fake_redis.subscribed()[0].endswith("{user-1/graph-1}/all")
    assert "user-1|graph#graph-1|executions" in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_a_failed_subscribe_leaves_no_bookkeeping(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    with (
        patch(
            "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
            side_effect=RedisConnectionError("shard full"),
        ),
        pytest.raises(RedisConnectionError),
    ):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id="graph-1", websocket=mock_websocket
        )

    assert connection_manager.subscriptions == {}
    assert connection_manager._ws_subs[mock_websocket] == {}
    assert connection_manager._ws_streams[mock_websocket] == {}


@pytest.mark.asyncio
async def test_unsubscribe_unknown_channel_returns_none(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Unsubscribing from a channel that was never subscribed must no-op."""
    result = await connection_manager.unsubscribe_graph_exec(
        user_id="user-1",
        graph_exec_id="never-subscribed",
        websocket=mock_websocket,
    )
    assert result is None


@pytest.mark.asyncio
async def test_disconnect_closes_every_stream_and_clears_bookkeeping(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    shards = [("redis", 17000), ("redis", 17001)]
    fake_redis.shard_of = lambda channel: shards[channel.endswith("b}/all")]
    await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    for graph_id in ("a", "b"):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=mock_websocket
        )

    await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")

    assert len(fake_redis.clients) == 2
    for client in fake_redis.clients:
        client.aclose.assert_awaited_once()
    assert mock_websocket not in connection_manager._ws_subs
    assert mock_websocket not in connection_manager._ws_streams
    assert mock_websocket not in connection_manager._ws_notifications
    assert connection_manager.subscriptions == {}
    assert connection_manager._user_sockets == {}


@pytest.mark.asyncio
async def test_a_lost_stream_closes_the_socket(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """Pub/sub keeps no history, so a socket whose stream broke is closed and
    the client reconnects, resubscribes and refetches what it missed."""
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-1", websocket=mock_websocket
    )

    fake_redis.pubsubs[0].inbox.put_nowait(RedisConnectionError("shard gone"))

    await _until(lambda: mock_websocket.close.await_count == 1)
    assert mock_websocket.close.await_args.kwargs["code"] == WS_CLOSE_EVENT_STREAM_LOST


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# ---------- _forward_exec_event ----------


@pytest.mark.asyncio
async def test_forward_exec_event_unwraps_wrapper_and_sends_ws_message(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Must strip the ``_EventPayloadWrapper`` envelope before re-sending."""
    channel_key = "user-1|graph_exec#x"
    inner = {
        "event_type": "node_execution_update",
        "user_id": "user-1",
        "graph_id": "g",
        "graph_version": 1,
        "graph_exec_id": "x",
        "node_exec_id": "n",
        "node_id": "nid",
        "block_id": "bid",
        "status": "COMPLETED",
    }
    wrapper = json.dumps({"payload": inner})

    await connection_manager._forward_exec_event(
        mock_websocket, channel_key, wrapper.encode()
    )

    mock_websocket.send_text.assert_called_once()
    sent = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent["method"] == WSMethod.NODE_EXECUTION_EVENT.value
    assert sent["channel"] == channel_key
    assert sent["data"]["graph_exec_id"] == "x"


@pytest.mark.asyncio
async def test_forward_exec_event_accepts_str_payload(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Decode path handles str payload identically to bytes."""
    inner = {
        "event_type": "graph_execution_update",
        "id": "x",
        "user_id": "user-1",
        "graph_id": "g",
        "graph_version": 1,
        "status": "COMPLETED",
    }
    wrapper = json.dumps({"payload": inner})
    await connection_manager._forward_exec_event(mock_websocket, "chan", wrapper)
    mock_websocket.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_forward_exec_event_noop_on_none_payload(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    await connection_manager._forward_exec_event(mock_websocket, "chan", None)
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_exec_event_drops_unknown_event_type(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Unknown event_type → silent drop, no WS send."""
    wrapper = json.dumps({"payload": {"event_type": "error_comms_update"}})
    await connection_manager._forward_exec_event(
        mock_websocket, "chan", wrapper.encode()
    )
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_exec_event_drops_non_dict_payload(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Bad envelope (payload not dict) must drop silently, no crash."""
    wrapper = json.dumps({"payload": "not-a-dict"})
    await connection_manager._forward_exec_event(
        mock_websocket, "chan", wrapper.encode()
    )
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_exec_event_swallows_malformed_json(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Garbage payload must not propagate — it just logs + drops."""
    await connection_manager._forward_exec_event(mock_websocket, "chan", b"not json")
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_exec_event_logs_ws_close_race_at_debug(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """SPUBLISH racing WS close → debug, not exception."""
    inner = {
        "event_type": "graph_execution_update",
        "id": "x",
        "user_id": "user-1",
        "graph_id": "g",
        "graph_version": 1,
        "status": "COMPLETED",
    }
    wrapper = json.dumps({"payload": inner})
    mock_websocket.send_text = AsyncMock(
        side_effect=RuntimeError(
            'Cannot call "send" once a close message has been sent.'
        )
    )
    with patch("backend.api.conn_manager.logger") as mock_logger:
        await connection_manager._forward_exec_event(
            mock_websocket, "chan", wrapper.encode()
        )
    mock_logger.exception.assert_not_called()
    mock_logger.debug.assert_called_once()
    assert "Dropped exec event on closed WS" in mock_logger.debug.call_args[0][0]


@pytest.mark.asyncio
async def test_forward_exec_event_logs_real_send_error_at_exception(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A non-close-race RuntimeError must still log at exception level."""
    inner = {
        "event_type": "graph_execution_update",
        "id": "x",
        "user_id": "user-1",
        "graph_id": "g",
        "graph_version": 1,
        "status": "COMPLETED",
    }
    wrapper = json.dumps({"payload": inner})
    mock_websocket.send_text = AsyncMock(side_effect=RuntimeError("something else"))
    # Force application_state to a sentinel that is NOT DISCONNECTED.
    mock_websocket.application_state = None
    mock_websocket.client_state = None
    with patch("backend.api.conn_manager.logger") as mock_logger:
        await connection_manager._forward_exec_event(
            mock_websocket, "chan", wrapper.encode()
        )
    mock_logger.exception.assert_called_once()


# ---------- _start_notification_subscription / _forward_notification ----------


@pytest.mark.asyncio
async def test_start_notification_subscription_stores_channel(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    await connection_manager._start_notification_subscription(
        mock_websocket, user_id="user-1"
    )
    channel = _notification_bus_channel("user-1")
    assert connection_manager._ws_notifications[mock_websocket] == channel
    assert fake_redis.subscribed() == [channel]


@pytest.mark.asyncio
async def test_start_notification_subscription_swallows_start_errors(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """A failed SSUBSCRIBE must not crash connect — the WS stays usable."""
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        side_effect=RuntimeError("SSUBSCRIBE failed"),
    ):
        await connection_manager._start_notification_subscription(
            mock_websocket, user_id="user-1"
        )
    # On failure the channel must NOT be stored (bookkeeping clean).
    assert mock_websocket not in connection_manager._ws_notifications


@pytest.mark.asyncio
async def test_forward_notification_delivers_to_owning_websocket(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """The notification wrapper → WSMessage round-trip must reach the owning WS."""
    inner = {
        "user_id": "user-1",
        "payload": {"type": "info", "event": "hi"},
    }
    wrapper = json.dumps({"payload": inner})

    await connection_manager._forward_notification(
        mock_websocket, "user-1", wrapper.encode()
    )

    mock_websocket.send_text.assert_called_once()
    sent = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent["method"] == WSMethod.NOTIFICATION.value
    assert sent["data"] == {"type": "info", "event": "hi"}


@pytest.mark.asyncio
async def test_forward_notification_rejects_cross_user_payload(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Defense in depth: a payload for a different user must be dropped."""
    inner = {
        "user_id": "user-2",
        "payload": {"type": "info", "event": "sneaky"},
    }
    wrapper = json.dumps({"payload": inner})

    await connection_manager._forward_notification(
        mock_websocket, "user-1", wrapper.encode()
    )

    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_notification_noop_on_none(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    await connection_manager._forward_notification(mock_websocket, "user-1", None)
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_notification_swallows_bad_envelope(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Malformed notification payload must not blow up or send anything."""
    await connection_manager._forward_notification(
        mock_websocket, "user-1", b"not json"
    )
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_notification_drops_non_dict_inner(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    wrapper = json.dumps({"payload": "not-a-dict"})
    await connection_manager._forward_notification(
        mock_websocket, "user-1", wrapper.encode()
    )
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_forward_notification_swallows_websocket_send_failure(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A flaky websocket must not propagate the failure to the pump."""
    mock_websocket.send_text = AsyncMock(side_effect=RuntimeError("ws closed"))
    inner = {
        "user_id": "user-1",
        "payload": {"type": "info", "event": "hi"},
    }
    wrapper = json.dumps({"payload": inner})
    # Must not raise.
    await connection_manager._forward_notification(
        mock_websocket, "user-1", wrapper.encode()
    )


@pytest.mark.asyncio
async def test_forward_notification_logs_ws_close_race_at_debug(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """SPUBLISH racing WS close on notification path → debug, not warning."""
    mock_websocket.send_text = AsyncMock(
        side_effect=RuntimeError(
            'Cannot call "send" once a close message has been sent.'
        )
    )
    inner = {
        "user_id": "user-1",
        "payload": {"type": "info", "event": "hi"},
    }
    wrapper = json.dumps({"payload": inner})
    with patch("backend.api.conn_manager.logger") as mock_logger:
        await connection_manager._forward_notification(
            mock_websocket, "user-1", wrapper.encode()
        )
    mock_logger.warning.assert_not_called()
    mock_logger.debug.assert_called_once()
    assert "Dropped notification on closed WS" in mock_logger.debug.call_args[0][0]


@pytest.mark.asyncio
async def test_connect_socket_starts_notification_subscription(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """connect_socket must call _start_notification_subscription once."""
    with patch.object(
        connection_manager, "_start_notification_subscription", AsyncMock()
    ) as mock_start:
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    mock_start.assert_awaited_once_with(mock_websocket, user_id="user-1")


def _execution_db(meta: GraphExecutionMeta | None) -> MagicMock:
    return MagicMock(get_graph_execution_meta=AsyncMock(return_value=meta))
