import ast
import asyncio
import gc
import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
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
from backend.api.ws_outbox import ENTRY_OVERHEAD_BYTES, WSBacklog, WSOutbox
from backend.data.execution import GraphExecutionMeta, graph_all_channel
from backend.data.sharded_pubsub import EventStreamLostError


class _FakePubSub:
    """redis-py's PubSub as ``ShardedSubscription`` drives it: ``ssubscribe``
    queues Redis's confirmation, and reads take from the inbox."""

    def __init__(self) -> None:
        self.ssubscribe = AsyncMock(side_effect=self._confirm)
        self.sunsubscribe = AsyncMock()
        self.ping = AsyncMock()
        self.aclose = AsyncMock()
        self.connection: MagicMock | None = None
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()

    async def connect(self) -> None:
        self.connection = MagicMock()

    async def _confirm(self, channel: str) -> None:
        self.inbox.put_nowait({"type": "ssubscribe", "channel": channel, "data": 1})

    async def get_message(self, timeout: float | None = None) -> Any:
        return await self.parse_response(block=timeout is None, timeout=timeout or 0)

    async def parse_response(self, block: bool = True, timeout: float = 0) -> Any:
        try:
            async with asyncio.timeout(None if block else timeout):
                item = await self.inbox.get()
        except TimeoutError:
            return None
        if isinstance(item, BaseException):
            raise item
        return item

    async def handle_message(self, response: Any) -> Any:
        return response


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
        patch("backend.api.conn_manager.redis.refresh_slot_map"),
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
async def outbox():
    """The socket's outbox a test's shard stream sends through."""
    box = WSOutbox(on_overflow=AsyncMock(), backlog=WSBacklog())
    yield box
    await box.stop()


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
    openers = {
        "ShardedSubscription.open",
        "listen_events",
        "listen_with_resync",
        "wait_for_event",
    }
    callers: set[str] = set()
    for path in api_dir.rglob("*.py"):
        if path.name.endswith("_test.py"):
            continue
        callers |= _callers_of(ast.parse(path.read_text()), openers, path.stem)
    assert callers == {"conn_manager._ShardStream._open_or_subscribe"}


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
                if name in names or ast.unparse(func) in names:
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
        # Accepted, then closed, so the browser sees the 1013 code.
        refused.accept.assert_awaited_once()
        refused.close.assert_awaited_once()
        assert refused.close.await_args.kwargs["code"] == WS_CLOSE_TRY_AGAIN_LATER

        # Another user is unaffected, and a closed socket frees its slot.
        assert await connection_manager.connect_socket(
            AsyncMock(spec=WebSocket), user_id="user-2"
        )
        await connection_manager.disconnect_socket(sockets[0], user_id="user-1")
        assert await connection_manager.connect_socket(refused, user_id="user-1")


@pytest.mark.asyncio
async def test_streams_per_shard_are_capped_per_process(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over all its sockets the process holds at most ``MAX_STREAMS_PER_SHARD``
    connections to a shard; at it, new sockets and new streams are refused."""
    monkeypatch.setattr("backend.api.conn_manager.MAX_STREAMS_PER_SHARD", 2)
    fake_redis.shard_of = lambda c: ("redis", 2) if c.endswith("/u-b") else ("redis", 1)
    ws1, ws2, ws3, ws4 = (AsyncMock(spec=WebSocket) for _ in range(4))

    assert await connection_manager.connect_socket(ws1, user_id="u-a")
    assert await connection_manager.connect_socket(ws2, user_id="u-b")
    assert await connection_manager.connect_socket(ws3, user_id="u-a")
    # Shard 1 now holds two streams (ws1's and ws3's notifications).
    with pytest.raises(ValueError, match="Server busy"):
        await connection_manager.subscribe_graph_execs(
            user_id="u-b", graph_id="g", websocket=ws2
        )
    assert not await connection_manager.connect_socket(ws4, user_id="u-c")
    assert ws4.close.await_args.kwargs["code"] == WS_CLOSE_TRY_AGAIN_LATER

    await connection_manager.disconnect_socket(ws1, user_id="u-a")
    assert await connection_manager.connect_socket(ws4, user_id="u-c")
    assert len(fake_redis.clients) == 4


@pytest.mark.asyncio
async def test_a_user_past_their_share_cannot_fill_a_shard(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.api.conn_manager.MAX_STREAMS_PER_SHARD", 4)
    monkeypatch.setattr("backend.api.conn_manager.STREAMS_PER_USER_SHARE", 1)
    heavy = [AsyncMock(spec=WebSocket) for _ in range(4)]
    for ws in heavy[:3]:
        assert await connection_manager.connect_socket(ws, user_id="u-a")
    # The shard's last stream is kept for users still within their share.
    assert not await connection_manager.connect_socket(heavy[3], user_id="u-a")
    assert await connection_manager.connect_socket(
        AsyncMock(spec=WebSocket), user_id="u-b"
    )

    # Disconnecting gives the heavy user its share back.
    for ws in heavy[:3]:
        await connection_manager.disconnect_socket(ws, user_id="u-a")
    for _ in range(2):
        assert await connection_manager.connect_socket(
            AsyncMock(spec=WebSocket), user_id="u-c"
        )
    assert await connection_manager.connect_socket(heavy[3], user_id="u-a")


@pytest.mark.asyncio
async def test_unsubscribing_a_streams_last_channel_frees_its_slot(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.api.conn_manager.MAX_STREAMS_PER_SHARD", 1)
    other: AsyncMock = AsyncMock(spec=WebSocket)
    with patch.object(
        connection_manager, "_start_notification_subscription", AsyncMock()
    ):
        assert await connection_manager.connect_socket(mock_websocket, user_id="u-a")
        await connection_manager.subscribe_graph_execs(
            user_id="u-a", graph_id="g", websocket=mock_websocket
        )
        assert not await connection_manager.connect_socket(other, user_id="u-b")

        await connection_manager.unsubscribe_graph_execs(
            user_id="u-a", graph_id="g", websocket=mock_websocket
        )
        assert await connection_manager.connect_socket(other, user_id="u-b")


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
    outbox: WSOutbox,
) -> None:
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/g}/exec/b", AsyncMock())
        assert len(fake_redis.clients) == 1
        assert fake_redis.subscribed() == ["{u/g}/exec/a", "{u/g}/exec/b"]
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_add_returns_once_redis_confirms_the_channel(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """The socket's answer to a subscribe means the channel is live: a view
    re-reads its state then, and an event published before Redis applied the
    SSUBSCRIBE would never arrive."""
    on_a = AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    adding: asyncio.Task[None] | None = None
    try:
        await stream.add("{u/g}/exec/a", on_a)
        pubsub = fake_redis.pubsubs[0]
        # The SSUBSCRIBE is written; Redis's confirmation comes later.
        pubsub.ssubscribe.side_effect = None
        adding = asyncio.create_task(stream.add("{u/g}/exec/b", AsyncMock()))
        await _until(lambda: pubsub.ssubscribe.await_count == 2)
        # Another channel's confirmation doesn't count. The pump has read it
        # once A's handler has the message queued behind it.
        pubsub.inbox.put_nowait(
            {"type": "ssubscribe", "channel": "{u/g}/exec/c", "data": 2}
        )
        pubsub.inbox.put_nowait(
            {"type": "smessage", "channel": "{u/g}/exec/a", "data": "marker"}
        )
        await _until(lambda: on_a.await_count == 1)
        assert not adding.done()

        pubsub.inbox.put_nowait(
            {"type": "ssubscribe", "channel": "{u/g}/exec/b", "data": 2}
        )
        await asyncio.wait_for(adding, timeout=5)
    finally:
        await _join(adding)
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_refuses_a_second_subscribe_of_a_channel_being_confirmed(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """ws_api handles a socket's messages one at a time, so this can't happen
    today; a caller that ever did it must fail at once, not wait 15 s."""
    first, second = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    adding: asyncio.Task[None] | None = None
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        adding = asyncio.create_task(stream.add("{u/g}/exec/b", first))
        await _until(lambda: pubsub.ssubscribe.await_count == 2)

        with pytest.raises(RuntimeError):
            await stream.add("{u/g}/exec/b", second)

        pubsub.inbox.put_nowait({"type": "ssubscribe", "channel": "{u/g}/exec/b"})
        await asyncio.wait_for(adding, timeout=5)
        assert stream.handlers["{u/g}/exec/b"] is first
    finally:
        await _join(adding)
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_delivers_a_message_right_behind_the_confirmation(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """The pump reads a buffered message right after the confirmation without
    pausing, so the channel's handler must be in place before it."""
    on_b = AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]

        async def confirm_then_publish(channel: str) -> None:
            pubsub.inbox.put_nowait({"type": "ssubscribe", "channel": channel})
            pubsub.inbox.put_nowait(
                {"type": "smessage", "channel": channel, "data": "first"}
            )

        pubsub.ssubscribe.side_effect = confirm_then_publish
        await asyncio.wait_for(stream.add("{u/g}/exec/b", on_b), timeout=5)

        await _until(lambda: on_b.await_count == 1)
        on_b.assert_awaited_once_with("first")
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_add_fails_at_once_once_the_stream_has_ended(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """Nothing reads a confirmation after the pump ends, and the socket is
    already closing; a subscribe then must not wait out the timeout."""
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        fake_redis.pubsubs[0].inbox.put_nowait(RedisConnectionError("shard gone"))
        await _until(lambda: on_lost.await_count == 1)

        with pytest.raises(EventStreamLostError):
            await asyncio.wait_for(stream.add("{u/g}/exec/b", AsyncMock()), 5)
        assert "{u/g}/exec/b" not in stream.handlers
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_add_fails_when_the_stream_is_lost_before_redis_confirms(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        adding = asyncio.create_task(stream.add("{u/g}/exec/b", AsyncMock()))
        await asyncio.sleep(0.05)
        pubsub.inbox.put_nowait(RedisConnectionError("shard gone"))

        with pytest.raises(EventStreamLostError):
            await asyncio.wait_for(adding, timeout=5)
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_add_times_out_when_redis_never_confirms(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        fake_redis.pubsubs[0].ssubscribe.side_effect = None

        with pytest.raises(TimeoutError):
            await stream.add("{u/g}/exec/b", AsyncMock())
        assert "{u/g}/exec/b" not in stream.handlers
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_subscribe_that_times_out_takes_its_channel_back(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    """Redis may apply an SSUBSCRIBE whose confirmation comes too late: the
    stream sends SUNSUBSCRIBE, so the channel doesn't stay subscribed with no
    handler, and its late confirmation and events change nothing."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    on_a, on_b, on_lost = AsyncMock(), AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", on_a)
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None

        with pytest.raises(TimeoutError):
            await stream.add("{u/g}/exec/b", on_b)
        pubsub.sunsubscribe.assert_awaited_once_with("{u/g}/exec/b")

        # What Redis sends next, in order: the late confirmation, an event,
        # the SUNSUBSCRIBE's confirmation, then an event for A.
        for message in (
            {"type": "ssubscribe", "channel": "{u/g}/exec/b"},
            {"type": "smessage", "channel": "{u/g}/exec/b", "data": "late"},
            {"type": "sunsubscribe", "channel": "{u/g}/exec/b"},
            {"type": "smessage", "channel": "{u/g}/exec/a", "data": "for-a"},
        ):
            pubsub.inbox.put_nowait(message)
        await _until(lambda: on_a.await_count == 1)
        on_a.assert_awaited_once_with("for-a")
        on_b.assert_not_awaited()
        on_lost.assert_not_awaited()
        assert stream._unsubscribing == {}
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_events_of_a_channel_being_taken_back_reach_no_handler(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    """The timed-out subscription's handler stays registered until its take
    back is written; an event Redis sent it meanwhile must not get through."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    monkeypatch.setattr("backend.api.conn_manager._TAKE_BACK_TIMEOUT_S", 60.0)
    on_a, on_b = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    written = asyncio.Event()
    adding: asyncio.Task[None] | None = None
    try:
        await stream.add("{u/g}/exec/a", on_a)
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None

        async def write_slowly(channel: str) -> None:
            await written.wait()

        pubsub.sunsubscribe.side_effect = write_slowly
        adding = asyncio.create_task(stream.add("{u/g}/exec/b", on_b))
        await _until(lambda: pubsub.sunsubscribe.await_count == 1)
        pubsub.inbox.put_nowait(
            {"type": "smessage", "channel": "{u/g}/exec/b", "data": "late"}
        )
        pubsub.inbox.put_nowait(
            {"type": "smessage", "channel": "{u/g}/exec/a", "data": "marker"}
        )
        await _until(lambda: on_a.await_count == 1)
        on_b.assert_not_awaited()

        written.set()
        done, _ = await asyncio.wait([adding], timeout=5)
        assert adding in done
        assert isinstance(adding.exception(), TimeoutError)
    finally:
        written.set()
        await _join(adding)
        await stream.stop()


@pytest.mark.asyncio
async def test_a_retry_after_a_take_back_waits_for_its_own_confirmation(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    """The late confirmation of the first SSUBSCRIBE, and any event sent to
    it, come before the take back's SUNSUBSCRIBE is answered, so neither can
    stand for the retry's confirmation or reach the retry's handler."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    on_a, retry_on_b = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    retrying: asyncio.Task[None] | None = None
    try:
        await stream.add("{u/g}/exec/a", on_a)
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        with pytest.raises(TimeoutError):
            await stream.add("{u/g}/exec/b", AsyncMock())
        # Ten times the waits below, so only a missing confirmation ends it.
        monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 60.0)
        retrying = asyncio.create_task(stream.add("{u/g}/exec/b", retry_on_b))
        await _until(lambda: pubsub.ssubscribe.await_count == 3)

        for message, markers in (
            ({"type": "ssubscribe", "channel": "{u/g}/exec/b"}, 1),
            ({"type": "smessage", "channel": "{u/g}/exec/b", "data": "old"}, 2),
            ({"type": "sunsubscribe", "channel": "{u/g}/exec/b"}, 3),
        ):
            pubsub.inbox.put_nowait(message)
            pubsub.inbox.put_nowait(
                {"type": "smessage", "channel": "{u/g}/exec/a", "data": "marker"}
            )
            await _until(lambda n=markers: on_a.await_count == n)
            assert not retrying.done()
        retry_on_b.assert_not_awaited()

        pubsub.inbox.put_nowait({"type": "ssubscribe", "channel": "{u/g}/exec/b"})
        await asyncio.wait_for(retrying, timeout=5)
        pubsub.inbox.put_nowait(
            {"type": "smessage", "channel": "{u/g}/exec/b", "data": "new"}
        )
        await _until(lambda: retry_on_b.await_count == 1)
        retry_on_b.assert_awaited_once_with("new")
    finally:
        await _join(retrying)
        await stream.stop()


@pytest.mark.asyncio
async def test_a_take_back_that_fails_retires_the_stream(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    """Without the SUNSUBSCRIBE, the channel may stay subscribed with no
    handler; closing the stream drops it, and the socket resubscribes."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        pubsub.sunsubscribe.side_effect = RedisConnectionError("shard gone")

        with pytest.raises(TimeoutError):
            await stream.add("{u/g}/exec/b", AsyncMock())
        on_lost.assert_awaited_once()
        fake_redis.clients[0].aclose.assert_awaited_once()
        assert stream._unsubscribing == {}
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_take_back_that_fails_unexpectedly_still_retires_the_stream(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    """Not only a lost stream or the deadline: any failure of the write leaves
    no confirmation coming, so the count must drop and the stream retire."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        pubsub.sunsubscribe.side_effect = RuntimeError("unexpected")

        with pytest.raises(TimeoutError):
            await stream.add("{u/g}/exec/b", AsyncMock())
        on_lost.assert_awaited_once()
        fake_redis.clients[0].aclose.assert_awaited_once()
        assert stream._unsubscribing == {}
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_take_back_that_never_finishes_retires_the_stream(
    fake_redis: _FakeRedis, monkeypatch: pytest.MonkeyPatch, outbox: WSOutbox
) -> None:
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.1)
    monkeypatch.setattr("backend.api.conn_manager._TAKE_BACK_TIMEOUT_S", 0.1)
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None

        async def never_written(channel: str) -> None:
            await asyncio.Event().wait()

        pubsub.sunsubscribe.side_effect = never_written
        # The outer bound turns a missing take-back deadline into a failure.
        async with asyncio.timeout(5):
            with pytest.raises(TimeoutError):
                await stream.add("{u/g}/exec/b", AsyncMock())
        on_lost.assert_awaited_once()
        fake_redis.clients[0].aclose.assert_awaited_once()
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_failed_write_leaves_no_unread_confirmation_error(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """The pump can fail the confirmation (the stream ended) while the
    SSUBSCRIBE write is still out, and then the write fails too."""
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unread: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unread.append(context))
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        pubsub = fake_redis.pubsubs[0]

        async def stream_ends_then_write_fails(channel: str) -> None:
            pubsub.inbox.put_nowait(RedisConnectionError("shard gone"))
            await _until(lambda: stream._pump_ended)
            raise RedisConnectionError("write failed")

        pubsub.ssubscribe.side_effect = stream_ends_then_write_fails
        with pytest.raises(EventStreamLostError):
            await stream.add("{u/g}/exec/b", AsyncMock())
        gc.collect()
        assert unread == []
    finally:
        loop.set_exception_handler(previous_handler)
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_rejects_wildcard_channel_before_redis(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    with pytest.raises(ValueError):
        await _ShardStream(on_lost=AsyncMock(), outbox=outbox).add(
            "{u/g}/exec/*", AsyncMock()
        )
    assert fake_redis.clients == []


@pytest.mark.asyncio
async def test_stream_forwards_smessages_to_their_channels_handler(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    on_a, on_b = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
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
async def test_stream_survives_a_failing_handler(
    fake_redis: _FakeRedis, outbox: WSOutbox
) -> None:
    handler = AsyncMock(side_effect=[RuntimeError("boom"), None])
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", handler)
        inbox = fake_redis.pubsubs[0].inbox
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/a", "data": "1"})
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/a", "data": "2"})
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
async def test_stream_reports_a_lost_stream(
    fake_redis: _FakeRedis, event: Any, outbox: WSOutbox
) -> None:
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        fake_redis.pubsubs[0].inbox.put_nowait(event)
        await _until(lambda: on_lost.await_count == 1)
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_stream_keeps_flowing_after_its_own_sunsubscribe(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    on_b = AsyncMock()
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/g}/exec/b", on_b)
        await stream.remove("{u/g}/exec/a")
        inbox = fake_redis.pubsubs[0].inbox
        inbox.put_nowait({"type": "sunsubscribe", "channel": "{u/g}/exec/a"})
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/b", "data": "1"})
        await _until(lambda: on_b.await_count == 1)
        on_lost.assert_not_awaited()
        fake_redis.pubsubs[0].sunsubscribe.assert_awaited_once_with("{u/g}/exec/a")
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_failed_sunsubscribe_does_not_hide_a_later_drop(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/h}/exec/b", AsyncMock())
        fake_redis.pubsubs[0].sunsubscribe.side_effect = RedisConnectionError("x")
        await stream.remove("{u/g}/exec/a")
        # Redis drops the channel itself later, e.g. after a slot move.
        fake_redis.pubsubs[0].inbox.put_nowait(
            {"type": "sunsubscribe", "channel": "{u/g}/exec/a"}
        )
        await _until(lambda: on_lost.await_count == 1)
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_unsubscribing_forgets_a_channel_once_its_sunsubscribe_confirms(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    """A long-lived stream must not keep a {channel: 0} entry per channel it
    has ever unsubscribed from."""
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/keep", AsyncMock())
        inbox = fake_redis.pubsubs[0].inbox
        for i in range(5):
            channel = f"{{u/g}}/exec/churn-{i}"
            await stream.add(channel, AsyncMock())
            await stream.remove(channel)
            inbox.put_nowait({"type": "sunsubscribe", "channel": channel})
            await _until(lambda c=channel: stream._unsubscribing[c] == 0)
        assert stream._unsubscribing == {}
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_a_quick_resubscribe_is_not_read_as_a_lost_stream(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    on_a = AsyncMock()
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    adding: asyncio.Task[None] | None = None
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())
        await stream.add("{u/g}/exec/b", AsyncMock())
        await stream.remove("{u/g}/exec/a")
        pubsub = fake_redis.pubsubs[0]
        pubsub.ssubscribe.side_effect = None
        adding = asyncio.create_task(stream.add("{u/g}/exec/a", on_a))
        await _until(lambda: pubsub.ssubscribe.await_count == 3)
        # The removal's confirmation arrives after the channel is back, and
        # Redis confirms the new SSUBSCRIBE after it, in the order sent.
        inbox = pubsub.inbox
        inbox.put_nowait({"type": "sunsubscribe", "channel": "{u/g}/exec/a"})
        inbox.put_nowait({"type": "ssubscribe", "channel": "{u/g}/exec/a"})
        await asyncio.wait_for(adding, timeout=5)
        inbox.put_nowait({"type": "smessage", "channel": "{u/g}/exec/a", "data": "1"})
        await _until(lambda: on_a.await_count == 1)
        on_lost.assert_not_awaited()
    finally:
        await _join(adding)
        await stream.stop()


@pytest.mark.asyncio
async def test_opening_a_stream_gives_up_when_redis_does_not_answer(
    monkeypatch: pytest.MonkeyPatch,
    outbox: WSOutbox,
) -> None:
    """The shard may have failed over, so giving up must refresh the slot map
    for the client's reconnect: a bound of the socket's own around the open
    would cancel it first. Shorter here, so that order is certain."""
    monkeypatch.setattr("backend.data.sharded_pubsub.OPEN_TIMEOUT_S", 0.2)
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.05)

    async def never_connects(channel: str) -> MagicMock:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    # The outer bound turns a missing open timeout into a failure, not a hang.
    async with asyncio.timeout(5):
        with (
            patch(
                "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
                side_effect=never_connects,
            ),
            patch("backend.api.conn_manager.redis.refresh_slot_map") as refresh,
            pytest.raises(EventStreamLostError),
        ):
            await stream.add("{u/g}/exec/a", AsyncMock())
    assert stream.handlers == {}
    refresh.assert_called_once()


@pytest.mark.asyncio
async def test_a_stalled_ssubscribe_does_not_hang_past_the_open_timeout(
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
    outbox: WSOutbox,
) -> None:
    """Opening's timeout must cover SSUBSCRIBE too, not just opening the
    connection, or a stalled subscribe holds the stream slot forever."""
    monkeypatch.setattr("backend.data.sharded_pubsub.OPEN_TIMEOUT_S", 0.05)
    original_connect = fake_redis.connect

    async def connect_then_stall(channel: str) -> MagicMock:
        client = await original_connect(channel)

        async def never_subscribes(*_args: Any, **_kwargs: Any) -> None:
            await asyncio.Event().wait()

        fake_redis.pubsubs[-1].ssubscribe.side_effect = never_subscribes
        return client

    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    # The outer bound turns a missing subscribe timeout into a failure, not a hang.
    async with asyncio.timeout(5):
        with (
            patch(
                "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
                side_effect=connect_then_stall,
            ),
            pytest.raises(EventStreamLostError),
        ):
            await stream.add("{u/g}/exec/a", AsyncMock())
    assert stream.handlers == {}
    fake_redis.clients[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stalled_ssubscribe_on_an_open_stream_gives_up(
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
    outbox: WSOutbox,
) -> None:
    """Adding a channel to an open stream has its own bound; the stream and
    its other channels stay as they were."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 0.05)
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/a", AsyncMock())

        async def never_subscribes(*_args: Any, **_kwargs: Any) -> None:
            await asyncio.Event().wait()

        fake_redis.pubsubs[0].ssubscribe.side_effect = never_subscribes
        async with asyncio.timeout(5):
            with pytest.raises(TimeoutError):
                await stream.add("{u/g}/exec/b", AsyncMock())
        assert list(stream.handlers) == ["{u/g}/exec/a"]
        fake_redis.clients[0].aclose.assert_not_awaited()
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_removing_the_last_channel_closes_the_connection(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    on_lost = AsyncMock()
    stream = _ShardStream(on_lost=on_lost, outbox=outbox)
    await stream.add("{u/g}/exec/a", AsyncMock())

    await stream.remove("{u/g}/exec/a")

    fake_redis.pubsubs[0].aclose.assert_awaited_once()
    fake_redis.clients[0].aclose.assert_awaited_once()
    on_lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_first_subscribe_closes_the_connection(
    fake_redis: _FakeRedis,
    outbox: WSOutbox,
) -> None:
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
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
        pytest.raises(EventStreamLostError),
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
        pytest.raises(EventStreamLostError),
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
    assert mock_websocket not in connection_manager._outboxes
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


async def _join(task: asyncio.Task[Any] | None) -> None:
    """Ends a task a test started, however the test went, and takes its
    outcome, so nothing is left to fail at teardown."""
    if task is None:
        return
    task.cancel()
    await asyncio.wait([task])
    if not task.cancelled():
        task.exception()


# ---------- a client that reads slowly (#16916) ----------

# Only a step that never comes keeps a wait going this long.
_HANG_SECONDS = 10.0


def _graph_event(graph_id: str) -> dict[str, Any]:
    """An event on a graph's runs channel, as Redis delivers it."""
    payload = {
        "event_type": "graph_execution_update",
        "id": f"exec-{graph_id}",
        "graph_id": graph_id,
        "status": "RUNNING",
    }
    return {
        "type": "smessage",
        "channel": event_bus_channel(graph_all_channel("user-1", graph_id)),
        "data": json.dumps({"payload": payload}),
    }


class _StuckClient:
    """A socket's client that has stopped reading: each send waits until
    ``reading`` is set, as uvicorn's does while the transport is paused."""

    def __init__(self, websocket: AsyncMock) -> None:
        self.reading = asyncio.Event()
        self.waited_on = asyncio.Event()
        self.cancelled = asyncio.Event()
        websocket.send_text = AsyncMock(side_effect=self._send)

    async def _send(self, message: str) -> None:
        self.waited_on.set()
        try:
            await self.reading.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_a_subscribe_is_confirmed_while_a_send_waits_on_the_client(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While an event's send waits on the client, the shard stream still
    reads Redis's confirmation of the next subscribe on it, so the subscribe
    is answered instead of waiting out its bound."""
    monkeypatch.setattr("backend.api.conn_manager._SUBSCRIBE_TIMEOUT_S", 60.0)
    client = _StuckClient(mock_websocket)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=mock_websocket
    )
    fake_redis.pubsubs[0].inbox.put_nowait(_graph_event("graph-a"))
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
            await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id="graph-b", websocket=mock_websocket
            )
        graph_b = event_bus_channel(graph_all_channel("user-1", "graph-b"))
        assert graph_b in fake_redis.subscribed()
        assert len(fake_redis.pubsubs) == 1
        # The first event's send, still waiting on the client.
        assert mock_websocket.send_text.await_count == 1
    finally:
        client.reading.set()


@pytest.mark.asyncio
async def test_a_socket_waiting_on_its_client_holds_up_no_other_socket(
    connection_manager: ConnectionManager, fake_redis: _FakeRedis
) -> None:
    stuck_ws: AsyncMock = AsyncMock(spec=WebSocket)
    stuck = _StuckClient(stuck_ws)
    healthy_ws: AsyncMock = AsyncMock(spec=WebSocket)
    healthy_ws.send_text = AsyncMock()
    for ws, graph_id in ((stuck_ws, "graph-a"), (healthy_ws, "graph-b")):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=ws
        )
    stuck_inbox, healthy_inbox = (p.inbox for p in fake_redis.pubsubs)
    try:
        stuck_inbox.put_nowait(_graph_event("graph-a"))
        async with asyncio.timeout(_HANG_SECONDS):
            await stuck.waited_on.wait()
        healthy_inbox.put_nowait(_graph_event("graph-b"))
        await _until(lambda: healthy_ws.send_text.await_count == 1)
    finally:
        stuck.reading.set()


@pytest.mark.asyncio
async def test_a_sockets_streams_on_every_shard_share_its_bound(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client is what falls behind, so its events count against one
    bound whichever shard they come from, and go out through one sender."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    fake_redis.shard_of = lambda channel: ("redis", 2 if "graph-b" in channel else 1)
    client = _StuckClient(mock_websocket)
    for graph_id in ("graph-a", "graph-b"):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=mock_websocket
        )
    first, second = (p.inbox for p in fake_redis.pubsubs)
    try:
        first.put_nowait(_graph_event("graph-a"))
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        second.put_nowait(_graph_event("graph-b"))
        await _until(lambda: mock_websocket.close.await_count == 1)
        mock_websocket.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
        assert mock_websocket.send_text.await_count == 1
    finally:
        client.reading.set()


@pytest.mark.asyncio
async def test_a_queued_event_of_a_dropped_subscription_never_reaches_the_client(
    fake_redis: _FakeRedis, outbox: WSOutbox
) -> None:
    """An event waits behind a slow send while the client unsubscribes and
    subscribes again: it belongs to the old subscription, and the new one's
    view has re-read its state since."""
    reading = asyncio.Event()

    async def wait_on_the_client(data: Any) -> None:
        await reading.wait()

    old, new = AsyncMock(), AsyncMock()
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    try:
        await stream.add("{u/g}/exec/slow", AsyncMock(side_effect=wait_on_the_client))
        await stream.add("{u/g}/exec/x", old)
        pubsub = fake_redis.pubsubs[0]
        for channel, data in (("slow", "first"), ("x", "stale")):
            pubsub.inbox.put_nowait(
                {"type": "smessage", "channel": f"{{u/g}}/exec/{channel}", "data": data}
            )
        # Its confirmation comes after both events, so they are queued now.
        await asyncio.wait_for(
            stream.add("{u/g}/exec/marker", AsyncMock()), _HANG_SECONDS
        )

        await stream.remove("{u/g}/exec/x")
        pubsub.inbox.put_nowait({"type": "sunsubscribe", "channel": "{u/g}/exec/x"})
        await asyncio.wait_for(stream.add("{u/g}/exec/x", new), _HANG_SECONDS)
        pubsub.inbox.put_nowait(
            {"type": "smessage", "channel": "{u/g}/exec/x", "data": "fresh"}
        )
        reading.set()

        await _until(lambda: new.await_count == 1)
        new.assert_awaited_once_with("fresh")
        old.assert_not_awaited()
    finally:
        reading.set()
        await stream.stop()


@pytest.mark.asyncio
async def test_a_socket_whose_client_falls_too_far_behind_is_closed(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Past the outbox's bound the client would miss events anyway, so it is
    closed as for a lost stream: it reconnects, resubscribes and refetches.
    The warning names the bound and how far behind the client was."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 3)
    client = _StuckClient(mock_websocket)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=mock_websocket
    )
    inbox = fake_redis.pubsubs[0].inbox
    inbox.put_nowait(_graph_event("graph-a"))
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        for _ in range(3):
            inbox.put_nowait(_graph_event("graph-a"))
        await _until(lambda: mock_websocket.close.await_count == 1)
        mock_websocket.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
        assert client.cancelled.is_set()
        assert "fell too far behind (MAX_QUEUED_EVENTS: 3 events," in caplog.text
    finally:
        client.reading.set()


@pytest.mark.asyncio
async def test_a_socket_closed_for_falling_behind_stops_reading_redis(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its shard streams would only drop what they read, and its close may
    never get through to a client that stopped reading, so they stop at once.
    Retired, it opens no stream on a shard it hadn't used either. Another
    socket's stream on the same shard reads on."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    fake_redis.shard_of = lambda channel: ("redis", 2 if "graph-c" in channel else 1)
    stuck_ws: AsyncMock = AsyncMock(spec=WebSocket)
    stuck = _StuckClient(stuck_ws)
    healthy_ws: AsyncMock = AsyncMock(spec=WebSocket)
    healthy_ws.send_text = AsyncMock()
    for ws, graph_id in ((stuck_ws, "graph-a"), (healthy_ws, "graph-b")):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=ws
        )
    stuck_redis, healthy_redis = fake_redis.clients
    stuck_inbox, healthy_inbox = (p.inbox for p in fake_redis.pubsubs)
    try:
        stuck_inbox.put_nowait(_graph_event("graph-a"))
        async with asyncio.timeout(_HANG_SECONDS):
            await stuck.waited_on.wait()
        stuck_inbox.put_nowait(_graph_event("graph-a"))
        await _until(lambda: stuck_ws.close.await_count == 1)
        stuck_redis.aclose.assert_awaited_once()
        with pytest.raises(EventStreamLostError):
            await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id="graph-c", websocket=stuck_ws
            )
        assert len(fake_redis.clients) == 2

        healthy_redis.aclose.assert_not_awaited()
        healthy_inbox.put_nowait(_graph_event("graph-b"))
        await _until(lambda: healthy_ws.send_text.await_count == 1)
    finally:
        stuck.reading.set()


async def _two_streams_one_slow_to_stop(
    connection_manager: ConnectionManager, fake_redis: _FakeRedis
) -> tuple[AsyncMock, _StuckClient, _ShardStream, asyncio.Event, asyncio.Event]:
    """A socket subscribed on two shards, "graph-a" and "graph-b", whose
    first stream's stop waits on ``release`` once ``stopping`` is set."""
    shard_of = {"graph-a": 1, "graph-b": 2, "graph-c": 3, "graph-d": 4}
    fake_redis.shard_of = lambda channel: next(
        ("redis", shard) for graph_id, shard in shard_of.items() if graph_id in channel
    )
    ws: AsyncMock = AsyncMock(spec=WebSocket)
    client = _StuckClient(ws)
    for graph_id in ("graph-a", "graph-b"):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=ws
        )
    first, _ = connection_manager._ws_streams[ws].values()
    stopping, release = asyncio.Event(), asyncio.Event()

    async def close_slowly() -> None:
        stopping.set()
        await release.wait()

    fake_redis.clients[0].aclose.side_effect = close_slowly
    return ws, client, first, stopping, release


async def _fall_behind(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    ws: AsyncMock,
    client: _StuckClient,
) -> None:
    """Past a bound of one event on the socket's first stream."""
    inbox = fake_redis.pubsubs[0].inbox
    inbox.put_nowait(_graph_event("graph-a"))
    async with asyncio.timeout(_HANG_SECONDS):
        await client.waited_on.wait()
    inbox.put_nowait(_graph_event("graph-a"))
    await _until(lambda: connection_manager._outboxes[ws]._closer is not None)


@pytest.mark.asyncio
async def test_a_slow_client_is_closed_while_its_socket_changes_streams(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close runs in its own task while the socket's messages are still
    handled. While one stream stops, the socket drops another and asks for a
    new shard: it is closed anyway with nothing left raised, the retired
    socket gets no new stream, and another socket still subscribes."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    ws, client, _, stopping, release = await _two_streams_one_slow_to_stop(
        connection_manager, fake_redis
    )
    healthy_ws: AsyncMock = AsyncMock(spec=WebSocket)
    healthy_ws.send_text = AsyncMock()
    try:
        await _fall_behind(connection_manager, fake_redis, ws, client)
        async with asyncio.timeout(_HANG_SECONDS):
            await stopping.wait()
        await connection_manager.unsubscribe_graph_execs(
            user_id="user-1", graph_id="graph-b", websocket=ws
        )
        with pytest.raises(EventStreamLostError):
            await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id="graph-c", websocket=ws
            )
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id="graph-d", websocket=healthy_ws
        )
        release.set()

        closer = connection_manager._outboxes[ws]._closer
        assert closer is not None
        async with asyncio.timeout(_HANG_SECONDS):
            await asyncio.wait([closer])
        assert closer.exception() is None
        ws.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
        # The retired socket's two streams, then the healthy socket's one.
        assert len(fake_redis.clients) == 3
        fake_redis.pubsubs[2].inbox.put_nowait(_graph_event("graph-d"))
        await _until(lambda: healthy_ws.send_text.await_count == 1)
    finally:
        release.set()
        client.reading.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["raises", "is-cancelled"])
async def test_a_slow_client_is_closed_whatever_a_stream_stop_does(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    """A stop that fails, or a close cancelled midway (as disconnecting the
    socket does), still sends the client its close."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    ws, client, first, stopping, release = await _two_streams_one_slow_to_stop(
        connection_manager, fake_redis
    )
    failing_stop = (
        patch.object(first, "stop", AsyncMock(side_effect=RuntimeError("boom")))
        if stop == "raises"
        else nullcontext()
    )
    try:
        with failing_stop:
            await _fall_behind(connection_manager, fake_redis, ws, client)
            closer = connection_manager._outboxes[ws]._closer
            assert closer is not None
            if stop == "is-cancelled":
                async with asyncio.timeout(_HANG_SECONDS):
                    await stopping.wait()
                closer.cancel()
            release.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([closer])
        ws.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
        if stop == "raises":
            assert closer.exception() is None
            # The stream after the failed one still stopped.
            fake_redis.clients[1].aclose.assert_awaited()
    finally:
        release.set()
        client.reading.set()


@pytest.mark.asyncio
async def test_a_stopped_stream_takes_no_new_subscribe(
    fake_redis: _FakeRedis, outbox: WSOutbox
) -> None:
    """Retired with its socket closing, it would otherwise reopen a Redis
    connection for a client that is gone."""
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    await stream.add("{u/g}/exec/a", AsyncMock())
    await stream.stop()
    with pytest.raises(EventStreamLostError):
        await stream.add("{u/g}/exec/b", AsyncMock())
    assert len(fake_redis.clients) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["opening", "confirmed"])
async def test_a_stream_stopped_while_a_subscribe_waits_keeps_nothing_running(
    fake_redis: _FakeRedis, outbox: WSOutbox, window: str
) -> None:
    """A slow client's close stops its streams from another task, so a stop
    can land while ``add()`` waits: on the first channel's connection, which
    it finds not there yet to close, or right after a later channel's
    confirmation. The subscribe fails, the connection is closed, and no pump
    is left running."""
    stream = _ShardStream(on_lost=AsyncMock(), outbox=outbox)
    stopping: list[asyncio.Task[None]] = []
    adding: asyncio.Task[None] | None = None
    try:
        if window == "opening":
            connecting, connected = asyncio.Event(), asyncio.Event()
            connect = fake_redis.connect

            async def connect_slowly(channel: str) -> MagicMock:
                connecting.set()
                await connected.wait()
                return await connect(channel)

            with patch(
                "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
                side_effect=connect_slowly,
            ):
                adding = asyncio.create_task(stream.add("{u/g}/exec/a", AsyncMock()))
                async with asyncio.timeout(_HANG_SECONDS):
                    await connecting.wait()
                    await stream.stop()
                connected.set()
                async with asyncio.timeout(_HANG_SECONDS):
                    await asyncio.wait([adding])
            channel = "{u/g}/exec/a"
        else:
            await stream.add("{u/g}/exec/a", AsyncMock())
            pubsub = fake_redis.pubsubs[0]

            async def confirm_then_stop(channel: str) -> None:
                pubsub.inbox.put_nowait({"type": "ssubscribe", "channel": channel})
                stopping.append(asyncio.create_task(stream.stop()))

            pubsub.ssubscribe.side_effect = confirm_then_stop
            adding = asyncio.create_task(stream.add("{u/g}/exec/b", AsyncMock()))
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([adding])
                # Started during the add; a failed stop fails the test.
                assert len(stopping) == 1
                await stopping[0]
            channel = "{u/g}/exec/b"
        assert isinstance(adding.exception(), EventStreamLostError)
        assert channel not in stream.handlers
        assert stream._task is None
        fake_redis.clients[0].aclose.assert_awaited()
    finally:
        await _join(adding)
        for task in stopping:
            await _join(task)
        await stream.stop()


@pytest.mark.asyncio
async def test_a_stream_stopped_while_it_opens_keeps_nothing_open(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subscribe can be opening the socket's stream on a new shard when the
    client falls behind and the socket's streams stop. That stop finds
    nothing to close yet, so the subscribe must close what it opened rather
    than start a pump on a retired stream."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    fake_redis.shard_of = lambda channel: ("redis", 2 if "graph-c" in channel else 1)
    ws: AsyncMock = AsyncMock(spec=WebSocket)
    client = _StuckClient(ws)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=ws
    )
    opening, opened = asyncio.Event(), asyncio.Event()
    connect = fake_redis.connect

    async def connect_slowly(channel: str) -> MagicMock:
        if "graph-c" in channel:
            opening.set()
            await opened.wait()
        return await connect(channel)

    subscribing: asyncio.Task[str] | None = None
    try:
        with patch(
            "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
            side_effect=connect_slowly,
        ):
            subscribing = asyncio.create_task(
                connection_manager.subscribe_graph_execs(
                    user_id="user-1", graph_id="graph-c", websocket=ws
                )
            )
            async with asyncio.timeout(_HANG_SECONDS):
                await opening.wait()
            await _fall_behind(connection_manager, fake_redis, ws, client)
            closer = connection_manager._outboxes[ws]._closer
            assert closer is not None
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([closer])
            ws.close.assert_awaited_once()
            opened.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([subscribing])
        assert isinstance(subscribing.exception(), EventStreamLostError)
        fake_redis.clients[-1].aclose.assert_awaited()
        assert ("redis", 2) not in connection_manager._ws_streams[ws]
    finally:
        opened.set()
        client.reading.set()
        await _join(subscribing)


@pytest.mark.asyncio
async def test_a_subscribe_that_opens_after_its_socket_retired_fails_at_once(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subscribe can finish opening a new shard's stream after its socket
    fell behind, while the close is still stopping another shard. It fails
    then and frees the shard's slot at once, rather than telling a closing
    socket it subscribed; the close goes out after."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    fake_redis.shard_of = lambda channel: ("redis", 2 if "graph-c" in channel else 1)
    ws: AsyncMock = AsyncMock(spec=WebSocket)
    client = _StuckClient(ws)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=ws
    )
    first_stopping, first_stopped = asyncio.Event(), asyncio.Event()

    async def close_first_slowly() -> None:
        first_stopping.set()
        await first_stopped.wait()

    fake_redis.clients[0].aclose.side_effect = close_first_slowly
    opening, opened = asyncio.Event(), asyncio.Event()
    connect = fake_redis.connect

    async def connect_slowly(channel: str) -> MagicMock:
        if "graph-c" in channel:
            opening.set()
            await opened.wait()
        return await connect(channel)

    subscribing: asyncio.Task[str] | None = None
    try:
        with patch(
            "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
            side_effect=connect_slowly,
        ):
            subscribing = asyncio.create_task(
                connection_manager.subscribe_graph_execs(
                    user_id="user-1", graph_id="graph-c", websocket=ws
                )
            )
            async with asyncio.timeout(_HANG_SECONDS):
                await opening.wait()
            second = connection_manager._ws_streams[ws][("redis", 2)]
            await _fall_behind(connection_manager, fake_redis, ws, client)
            async with asyncio.timeout(_HANG_SECONDS):
                await first_stopping.wait()
            opened.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([subscribing])
            assert isinstance(subscribing.exception(), EventStreamLostError)
            # Freed while the close still waits on the first shard.
            assert ("redis", 2) not in connection_manager._ws_streams[ws]
            assert ("redis", 2) not in connection_manager._streams_per_shard
            assert second._task is None
            fake_redis.clients[1].aclose.assert_awaited()
            ws.close.assert_not_awaited()

            first_stopped.set()
            closer = connection_manager._outboxes[ws]._closer
            assert closer is not None
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.wait([closer])
        assert closer.exception() is None
        ws.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
    finally:
        opened.set()
        first_stopped.set()
        client.reading.set()
        await _join(subscribing)


@pytest.mark.asyncio
async def test_a_full_process_closes_the_socket_furthest_behind(
    connection_manager: ConnectionManager,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every socket's outbox counts against one budget for the process, so
    sockets stuck behind can't add up past it: the one furthest behind is
    closed to make room for another socket's event."""
    stuck_ws: AsyncMock = AsyncMock(spec=WebSocket)
    stuck = _StuckClient(stuck_ws)
    healthy_ws: AsyncMock = AsyncMock(spec=WebSocket)
    healthy_ws.send_text = AsyncMock()
    for ws, graph_id in ((stuck_ws, "graph-a"), (healthy_ws, "graph-b")):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id=graph_id, websocket=ws
        )
    stuck_inbox, healthy_inbox = (p.inbox for p in fake_redis.pubsubs)
    # What each queued event counts: its payload and the queue entry itself.
    held = 2 * (sys.getsizeof(_graph_event("graph-a")["data"]) + ENTRY_OVERHEAD_BYTES)
    needed = sys.getsizeof(_graph_event("graph-b")["data"]) + ENTRY_OVERHEAD_BYTES
    monkeypatch.setattr(
        "backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", held + needed - 1
    )
    try:
        stuck_inbox.put_nowait(_graph_event("graph-a"))
        async with asyncio.timeout(_HANG_SECONDS):
            await stuck.waited_on.wait()
        stuck_inbox.put_nowait(_graph_event("graph-a"))
        await _until(lambda: connection_manager._backlog.queued_bytes == held)

        healthy_inbox.put_nowait(_graph_event("graph-b"))
        await _until(lambda: stuck_ws.close.await_count == 1)
        stuck_ws.close.assert_awaited_once_with(
            code=WS_CLOSE_EVENT_STREAM_LOST, reason="Client fell behind"
        )
        await _until(lambda: healthy_ws.send_text.await_count == 1)
        healthy_ws.close.assert_not_awaited()
    finally:
        stuck.reading.set()


@pytest.mark.asyncio
async def test_a_disconnect_cancelled_mid_stop_still_frees_the_sockets_backlog(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """At shutdown a socket's handler is cancelled, maybe while one of its
    streams is stopping; its outbox must stop anyway, so its backlog stops
    counting against the process and its send stops waiting."""
    client = _StuckClient(mock_websocket)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=mock_websocket
    )
    fake_redis.pubsubs[0].inbox.put_nowait(_graph_event("graph-a"))
    stopping, release = asyncio.Event(), asyncio.Event()

    async def close_slowly() -> None:
        stopping.set()
        await release.wait()

    fake_redis.clients[0].aclose.side_effect = close_slowly
    disconnecting: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        assert connection_manager._backlog.queued_bytes > 0
        disconnecting = asyncio.create_task(
            connection_manager.disconnect_socket(mock_websocket, user_id="user-1")
        )
        async with asyncio.timeout(_HANG_SECONDS):
            await stopping.wait()
        disconnecting.cancel()
        async with asyncio.timeout(_HANG_SECONDS):
            await asyncio.wait([disconnecting])
        assert disconnecting.cancelled()
        assert connection_manager._backlog.queued_bytes == 0
        assert mock_websocket not in connection_manager._outboxes
        assert client.cancelled.is_set()
    finally:
        release.set()
        client.reading.set()
        await _join(disconnecting)


@pytest.mark.asyncio
async def test_disconnect_ends_a_send_waiting_on_the_client(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    client = _StuckClient(mock_websocket)
    await connection_manager.subscribe_graph_execs(
        user_id="user-1", graph_id="graph-a", websocket=mock_websocket
    )
    fake_redis.pubsubs[0].inbox.put_nowait(_graph_event("graph-a"))
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
            await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")
        assert client.cancelled.is_set()
        assert mock_websocket not in connection_manager._outboxes
    finally:
        client.reading.set()


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
async def test_a_socket_whose_notification_stream_fails_to_open_is_closed(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
) -> None:
    """Kept open, it would never receive a notification and nothing would
    retry; closed with 1011, the client reconnects and refetches."""
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        side_effect=RedisConnectionError("shard gone before the confirmation"),
    ):
        accepted = await connection_manager.connect_socket(
            mock_websocket, user_id="user-1"
        )

    assert accepted is False
    mock_websocket.close.assert_awaited_once_with(
        code=WS_CLOSE_EVENT_STREAM_LOST, reason="Event stream interrupted"
    )
    assert mock_websocket not in connection_manager.active_connections
    assert mock_websocket not in connection_manager._ws_notifications
    assert mock_websocket not in connection_manager._ws_streams
    assert "user-1" not in connection_manager._user_sockets


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


@pytest.mark.asyncio
async def test_connect_socket_closes_when_notifications_lose_a_capacity_race(
    connection_manager: ConnectionManager,
    mock_websocket: AsyncMock,
    fake_redis: _FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission passes because the notification channel's shard isn't
    tracked yet, but capacity is gone by the time the subscribe itself runs
    (e.g. a concurrent subscriber took the last slot). The socket must be
    torn down, not handed back silently missing its notifications."""
    monkeypatch.setattr(
        ConnectionManager, "_has_room", lambda self, user_id, shard: False
    )

    assert not await connection_manager.connect_socket(mock_websocket, user_id="user-1")

    mock_websocket.accept.assert_awaited_once()
    mock_websocket.close.assert_awaited_once()
    assert mock_websocket.close.await_args.kwargs["code"] == WS_CLOSE_TRY_AGAIN_LATER
    assert "user-1" not in connection_manager._user_sockets
    assert mock_websocket not in connection_manager.active_connections


def _execution_db(meta: GraphExecutionMeta | None) -> MagicMock:
    return MagicMock(get_graph_execution_meta=AsyncMock(return_value=meta))
