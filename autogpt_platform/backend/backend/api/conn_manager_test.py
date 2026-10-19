import asyncio
import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from redis.exceptions import MovedError

from backend.api.conn_manager import (
    _MAX_CONCURRENT_STREAM_OPENS,
    WS_CLOSE_EVENT_STREAM_LOST,
    WS_CLOSE_EVENT_STREAM_LOST_REASON,
    WS_CLOSE_TRY_AGAIN_LATER,
    ConnectionManager,
    _graph_execs_channel_key,
    _notification_bus_channel,
    _Subscription,
    event_bus_channel,
    graph_exec_channel_key,
)
from backend.api.model import WSMethod
from backend.data.execution import GraphExecutionMeta


@pytest.fixture
def connection_manager() -> ConnectionManager:
    return ConnectionManager()


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
        connection_manager,
        "_start_notification_subscription",
        AsyncMock(return_value=True),
    ):
        assert await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    assert mock_websocket in connection_manager.active_connections
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_connect_refuses_the_handshake_when_the_stream_cannot_open(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """No socket is established that can't deliver events: the handshake is
    refused (try again later), so the client's backoff keeps growing instead
    of resetting on an "open" socket that closes straight away."""
    with patch.object(
        connection_manager,
        "_start_notification_subscription",
        AsyncMock(return_value=False),
    ):
        assert not await connection_manager.connect_socket(
            mock_websocket, user_id="user-1"
        )
    mock_websocket.accept.assert_not_called()
    mock_websocket.close.assert_awaited_once_with(
        code=WS_CLOSE_TRY_AGAIN_LATER, reason="Event stream unavailable"
    )
    assert mock_websocket not in connection_manager.active_connections


@pytest.mark.asyncio
async def test_connect_refuses_handshakes_beyond_the_admission_cap(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    start = AsyncMock(return_value=True)
    connection_manager._opening = _MAX_CONCURRENT_STREAM_OPENS
    with patch.object(connection_manager, "_start_notification_subscription", start):
        assert not await connection_manager.connect_socket(
            mock_websocket, user_id="user-1"
        )
    start.assert_not_awaited()
    mock_websocket.accept.assert_not_called()
    mock_websocket.close.assert_awaited_once_with(
        code=WS_CLOSE_TRY_AGAIN_LATER, reason="Server busy"
    )


@pytest.mark.asyncio
async def test_disconnect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.active_connections.add(mock_websocket)
    connection_manager.subscriptions["test_channel_42"] = {mock_websocket}

    await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")

    assert mock_websocket not in connection_manager.active_connections
    assert "test_channel_42" not in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_subscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock()
    fake_sub.stop = AsyncMock()
    with (
        patch(
            "backend.api.conn_manager.get_graph_execution_meta",
            AsyncMock(return_value=_meta()),
        ),
        patch("backend.api.conn_manager._Subscription", return_value=fake_sub),
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
    fake_sub.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_unsubscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    channel_key = "user-1|graph_exec#graph-exec-1"
    fake_sub = MagicMock()
    fake_sub.stop = AsyncMock()
    connection_manager.subscriptions[channel_key] = {mock_websocket}
    connection_manager._ws_subs[mock_websocket] = {channel_key: fake_sub}

    await connection_manager.unsubscribe_graph_exec(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        websocket=mock_websocket,
    )

    assert channel_key not in connection_manager.subscriptions
    fake_sub.stop.assert_awaited_once()


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


# ---------- _Subscription lifecycle ----------


def _fake_redis(listen) -> tuple[MagicMock, MagicMock]:
    """A shard-pinned client whose pubsub iterates ``listen()``."""
    fake_pubsub = MagicMock()
    fake_pubsub.ssubscribe = AsyncMock()
    fake_pubsub.sunsubscribe = AsyncMock()
    fake_pubsub.aclose = AsyncMock()
    fake_pubsub.listen = listen
    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock()
    return fake_client, fake_pubsub


async def _listen_forever():
    await asyncio.sleep(10)
    if False:
        yield


async def _run_pump_until_lost(listen, on_message=None) -> list[str]:
    """Start a subscription over ``listen`` and return the causes reported
    to ``on_lost`` once the pump has finished."""
    fake_client, fake_pubsub = _fake_redis(listen)
    lost: list[str] = []

    async def on_lost(cause: str) -> None:
        lost.append(cause)

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.subscribe_sharded_async",
        AsyncMock(return_value=(fake_client, fake_pubsub)),
    ):
        await sub.start(on_message or AsyncMock(), on_lost)
        assert sub._task is not None
        await asyncio.wait_for(sub._task, timeout=2.0)
        await sub.stop()
    return lost


@pytest.mark.asyncio
async def test_subscription_start_opens_a_confirmed_shard_subscription() -> None:
    """start() opens through subscribe_sharded_async (confirmed SSUBSCRIBE, no
    silent reconnects); stop() unsubscribes and closes both handles."""
    fake_client, fake_pubsub = _fake_redis(_listen_forever)
    subscribe = AsyncMock(return_value=(fake_client, fake_pubsub))

    sub = _Subscription("{u/g}/exec/x")
    with patch("backend.api.conn_manager.redis.subscribe_sharded_async", subscribe):
        await sub.start(AsyncMock(), AsyncMock())
        subscribe.assert_awaited_once_with("{u/g}/exec/x")
        await sub.stop()
    fake_pubsub.sunsubscribe.assert_awaited_once_with("{u/g}/exec/x")
    fake_pubsub.aclose.assert_awaited_once()
    fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_subscription_start_times_out_and_cleans_up() -> None:
    """Opening the stream retries for minutes while Redis is down; start()
    must give up after the open timeout instead of hanging the socket."""

    async def _hang(_channel: str):
        await asyncio.sleep(10)

    sub = _Subscription("{u/g}/exec/x")
    with (
        patch("backend.api.conn_manager.redis.subscribe_sharded_async", _hang),
        patch("backend.api.conn_manager._SUBSCRIBE_OPEN_TIMEOUT_S", 0.05),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await sub.start(AsyncMock(), AsyncMock())
    assert sub._task is None


@pytest.mark.asyncio
async def test_subscription_rejects_wildcard_channel_before_redis() -> None:
    """Constructor must guard against wildcard channels — SSUBSCRIBE can't pattern."""
    with pytest.raises(ValueError):
        _Subscription("{u/g}/exec/*")


@pytest.mark.asyncio
async def test_subscription_pump_ignores_non_message_types_and_forwards_smessage() -> (
    None
):
    """_pump() must skip subscribe confirmations and forward ``smessage`` payloads."""

    async def _listen():
        yield {"type": "ssubscribe", "data": 1}
        yield {"type": "smessage", "data": b'{"payload":"one"}'}
        yield {"type": "psubscribe", "data": 1}
        yield {"type": "smessage", "data": b'{"payload":"two"}'}

    received: list = []

    async def on_message(data) -> None:
        received.append(data)

    await _run_pump_until_lost(_listen, on_message)
    assert received == [b'{"payload":"one"}', b'{"payload":"two"}']


@pytest.mark.asyncio
async def test_subscription_pump_swallows_handler_exceptions() -> None:
    """A raising handler must not crash the pump — next message still delivers."""

    async def _listen():
        yield {"type": "smessage", "data": b"boom"}
        yield {"type": "smessage", "data": b"ok"}

    calls: list = []

    async def on_message(data) -> None:
        calls.append(data)
        if data == b"boom":
            raise RuntimeError("handler failed")

    await _run_pump_until_lost(_listen, on_message)
    assert calls == [b"boom", b"ok"]


@pytest.mark.asyncio
async def test_subscription_pump_reports_redis_errors_as_lost() -> None:
    """A dropped shard connection must be reported, not swallowed: the socket
    owner closes the WebSocket so the client reconnects and resyncs."""

    async def _listen():
        raise ConnectionError("Connection refused")
        yield  # pragma: no cover — unreachable

    assert await _run_pump_until_lost(_listen) == ["redis_error"]


@pytest.mark.asyncio
async def test_subscription_pump_reports_moved_errors_as_lost() -> None:
    async def _listen():
        raise MovedError("1234 127.0.0.1:7001")
        yield  # pragma: no cover — unreachable

    assert await _run_pump_until_lost(_listen) == ["redis_error"]


@pytest.mark.asyncio
async def test_subscription_pump_reports_server_sunsubscribe_as_lost() -> None:
    """A server-pushed sunsubscribe (slot moved) ends delivery on this shard."""

    async def _listen():
        yield {"type": "sunsubscribe", "data": 0}
        yield {"type": "smessage", "data": b"never forwarded"}

    on_message = AsyncMock()
    assert await _run_pump_until_lost(_listen, on_message) == ["revoked"]
    on_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_pump_reports_listen_end_and_crashes_as_lost() -> None:
    async def _ends():
        yield {"type": "ssubscribe", "data": 1}

    async def _crashes():
        raise RuntimeError("listen() exploded")
        yield  # pragma: no cover — unreachable

    assert await _run_pump_until_lost(_ends) == ["ended"]
    assert await _run_pump_until_lost(_crashes) == ["crashed"]


@pytest.mark.asyncio
async def test_subscription_stop_does_not_report_lost() -> None:
    """A deliberate stop() cancels the pump without closing the socket."""
    fake_client, fake_pubsub = _fake_redis(_listen_forever)
    on_lost = AsyncMock()

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.subscribe_sharded_async",
        AsyncMock(return_value=(fake_client, fake_pubsub)),
    ):
        await sub.start(AsyncMock(), on_lost)
        await sub.stop()
    on_lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_pump_noop_when_pubsub_missing() -> None:
    """Defensive: _pump returns fast if _pubsub was cleared before scheduling."""
    sub = _Subscription("{u/g}/exec/x")
    on_lost = AsyncMock()
    await sub._pump(AsyncMock(), on_lost)
    on_lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_stop_swallows_sunsubscribe_errors() -> None:
    """A broken cluster must not break graceful teardown."""
    fake_client, fake_pubsub = _fake_redis(_listen_forever)
    fake_pubsub.sunsubscribe = AsyncMock(side_effect=RuntimeError("SUNSUB broke"))
    fake_pubsub.aclose = AsyncMock(side_effect=RuntimeError("close broke"))
    fake_client.aclose = AsyncMock(side_effect=RuntimeError("client close broke"))

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.subscribe_sharded_async",
        AsyncMock(return_value=(fake_client, fake_pubsub)),
    ):
        await sub.start(AsyncMock(), AsyncMock())
        # Must not raise even if every teardown call blows up.
        await sub.stop()


# ---------- ConnectionManager closes sockets whose stream was lost ----------


@pytest.mark.asyncio
async def test_lost_stream_closes_socket_once(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """Every subscription on a socket can die together (one shard down);
    the socket is closed once, with the code the client reconnects on."""
    await connection_manager.close_for_lost_stream(mock_websocket, cause="redis_error")
    await connection_manager.close_for_lost_stream(mock_websocket, cause="revoked")

    mock_websocket.close.assert_awaited_once_with(
        code=WS_CLOSE_EVENT_STREAM_LOST, reason=WS_CLOSE_EVENT_STREAM_LOST_REASON
    )


@pytest.mark.asyncio
async def test_exec_subscription_loss_closes_its_socket(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    captured: dict = {}

    def _factory(full_channel: str) -> MagicMock:
        sub = MagicMock()

        async def _start(on_message, on_lost) -> None:
            captured["on_lost"] = on_lost

        sub.start = _start
        sub.stop = AsyncMock()
        return sub

    with patch("backend.api.conn_manager._Subscription", side_effect=_factory):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id="graph-1", websocket=mock_websocket
        )

    await captured["on_lost"]("redis_error")
    mock_websocket.close.assert_awaited_once_with(
        code=WS_CLOSE_EVENT_STREAM_LOST, reason=WS_CLOSE_EVENT_STREAM_LOST_REASON
    )


@pytest.mark.asyncio
async def test_failed_exec_subscribe_leaves_no_bookkeeping(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A subscribe whose stream never opened must not look subscribed."""
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock(side_effect=asyncio.TimeoutError())

    with patch("backend.api.conn_manager._Subscription", return_value=fake_sub):
        with pytest.raises(asyncio.TimeoutError):
            await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id="graph-1", websocket=mock_websocket
            )

    assert connection_manager.subscriptions == {}
    assert connection_manager._ws_subs.get(mock_websocket, {}) == {}


# ---------- ConnectionManager subscribe reuse / missing-meta handling ----------


@pytest.mark.asyncio
async def test_subscribe_graph_exec_raises_when_meta_missing(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """If graph_id can't be resolved, subscribe must reject — no silent SSUBSCRIBE."""
    with patch(
        "backend.api.conn_manager.get_graph_execution_meta",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(ValueError):
            await connection_manager.subscribe_graph_exec(
                user_id="user-1",
                graph_exec_id="unknown-exec",
                websocket=mock_websocket,
            )


@pytest.mark.asyncio
async def test_subscribe_graph_exec_reuses_existing_subscription(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A second subscribe on the same (ws, channel) must not open a second pump."""
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock()
    fake_sub.stop = AsyncMock()

    with (
        patch(
            "backend.api.conn_manager.get_graph_execution_meta",
            AsyncMock(return_value=_meta()),
        ),
        patch("backend.api.conn_manager._Subscription", return_value=fake_sub),
    ):
        await connection_manager.subscribe_graph_exec(
            user_id="user-1",
            graph_exec_id="graph-exec-1",
            websocket=mock_websocket,
        )
        await connection_manager.subscribe_graph_exec(
            user_id="user-1",
            graph_exec_id="graph-exec-1",
            websocket=mock_websocket,
        )

    fake_sub.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_subscribe_graph_execs_opens_aggregate_channel(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """subscribe_graph_execs must SSUBSCRIBE on the aggregate ``/all`` channel."""
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock()

    captured: list[str] = []

    def _capture(full_channel: str) -> MagicMock:
        captured.append(full_channel)
        return fake_sub

    with patch("backend.api.conn_manager._Subscription", side_effect=_capture):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1",
            graph_id="graph-1",
            websocket=mock_websocket,
        )

    assert captured, "must construct a Subscription"
    # Aggregate channel uses {user_id/graph_id} hash tag + /all suffix.
    assert captured[0].endswith("{user-1/graph-1}/all")
    assert "user-1|graph#graph-1|executions" in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_unsubscribe_graph_execs_tears_down(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """unsubscribe_graph_execs must drop bookkeeping and stop the subscription."""
    channel_key = "user-1|graph#graph-1|executions"
    fake_sub = MagicMock()
    fake_sub.stop = AsyncMock()
    connection_manager.subscriptions[channel_key] = {mock_websocket}
    connection_manager._ws_subs[mock_websocket] = {channel_key: fake_sub}

    await connection_manager.unsubscribe_graph_execs(
        user_id="user-1",
        graph_id="graph-1",
        websocket=mock_websocket,
    )

    assert channel_key not in connection_manager.subscriptions
    fake_sub.stop.assert_awaited_once()


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
async def test_disconnect_stops_all_pumps_and_notification_sub(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """disconnect must await stop() on every exec sub + the notification sub."""
    exec_sub_a = MagicMock()
    exec_sub_a.stop = AsyncMock()
    exec_sub_b = MagicMock()
    exec_sub_b.stop = AsyncMock()
    notif_sub = MagicMock()
    notif_sub.stop = AsyncMock()

    channel_a = "user-1|graph_exec#a"
    channel_b = "user-1|graph_exec#b"

    connection_manager.active_connections.add(mock_websocket)
    connection_manager._ws_subs[mock_websocket] = {
        channel_a: exec_sub_a,
        channel_b: exec_sub_b,
    }
    connection_manager._ws_notifications[mock_websocket] = notif_sub
    connection_manager.subscriptions[channel_a] = {mock_websocket}
    connection_manager.subscriptions[channel_b] = {mock_websocket}

    await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")

    exec_sub_a.stop.assert_awaited_once()
    exec_sub_b.stop.assert_awaited_once()
    notif_sub.stop.assert_awaited_once()
    assert mock_websocket not in connection_manager._ws_subs
    assert mock_websocket not in connection_manager._ws_notifications
    assert channel_a not in connection_manager.subscriptions
    assert channel_b not in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_two_connections_get_independent_subscriptions(
    connection_manager: ConnectionManager,
) -> None:
    """Each WS must get its own _Subscription (its own SSUBSCRIBE)."""
    ws_a: AsyncMock = AsyncMock(spec=WebSocket)
    ws_b: AsyncMock = AsyncMock(spec=WebSocket)
    calls: list[MagicMock] = []

    def _factory(full_channel: str):
        m = MagicMock()
        m.start = AsyncMock()
        m.stop = AsyncMock()
        calls.append(m)
        return m

    with (
        patch(
            "backend.api.conn_manager.get_graph_execution_meta",
            AsyncMock(return_value=_meta()),
        ),
        patch("backend.api.conn_manager._Subscription", side_effect=_factory),
    ):
        await connection_manager.subscribe_graph_exec(
            user_id="user-1", graph_exec_id="graph-exec-1", websocket=ws_a
        )
        await connection_manager.subscribe_graph_exec(
            user_id="user-1", graph_exec_id="graph-exec-1", websocket=ws_b
        )

    assert len(calls) == 2
    assert calls[0] is not calls[1]
    # Both websockets are on the same channel set.
    subs = connection_manager.subscriptions["user-1|graph_exec#graph-exec-1"]
    assert ws_a in subs and ws_b in subs


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
async def test_start_notification_subscription_stores_sub(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock()
    with patch("backend.api.conn_manager._Subscription", return_value=fake_sub):
        await connection_manager._start_notification_subscription(
            mock_websocket, user_id="user-1"
        )
    assert connection_manager._ws_notifications[mock_websocket] is fake_sub
    fake_sub.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_notification_subscription_swallows_start_errors(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A failed SSUBSCRIBE reports failure (connect_socket then refuses the
    handshake) instead of raising."""
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock(side_effect=RuntimeError("SSUBSCRIBE failed"))
    with patch("backend.api.conn_manager._Subscription", return_value=fake_sub):
        opened = await connection_manager._start_notification_subscription(
            mock_websocket, user_id="user-1"
        )
    assert opened is False
    # On failure the sub must NOT be stored (bookkeeping clean).
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
        connection_manager,
        "_start_notification_subscription",
        AsyncMock(return_value=True),
    ) as mock_start:
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    mock_start.assert_awaited_once_with(mock_websocket, user_id="user-1", accepted=ANY)


@pytest.mark.asyncio
async def test_a_notification_arriving_before_accept_waits_for_it(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """The notification stream opens before the handshake is accepted; a
    notification arriving while accept() is in flight is delivered after it,
    not sent on a socket that is still connecting (and dropped)."""
    captured: dict = {}
    order: list[str] = []
    finish_accept = asyncio.Event()

    def _factory(full_channel: str) -> MagicMock:
        sub = MagicMock()

        async def _start(on_message, on_lost) -> None:
            captured["on_message"] = on_message

        sub.start = _start
        sub.stop = AsyncMock()
        return sub

    async def _accept() -> None:
        await finish_accept.wait()
        order.append("accept")

    async def _forward(websocket, user_id, data) -> None:
        order.append("forward")

    mock_websocket.accept = AsyncMock(side_effect=_accept)
    with (
        patch("backend.api.conn_manager._Subscription", side_effect=_factory),
        patch.object(connection_manager, "_forward_notification", _forward),
    ):
        connect = asyncio.create_task(
            connection_manager.connect_socket(mock_websocket, user_id="user-1")
        )
        while mock_websocket.accept.await_count == 0:
            await asyncio.sleep(0)
        # accept() is now in flight; a notification arrives.
        delivered = asyncio.create_task(captured["on_message"](b"{}"))
        for _ in range(10):
            await asyncio.sleep(0)
        finish_accept.set()
        assert await connect
        await asyncio.wait_for(delivered, timeout=1)

    assert order == ["accept", "forward"]


def _gated_subscription_factory(captured: list[dict], release: asyncio.Event):
    """Fake subscriptions whose stop() blocks on ``release`` (so teardown
    can be interleaved) and that expose their on_lost callback."""

    def _factory(full_channel: str) -> MagicMock:
        entry: dict = {}
        sub = MagicMock()

        async def _start(on_message, on_lost) -> None:
            entry["on_lost"] = on_lost

        async def _stop() -> None:
            entry["stop_started"] = True
            await release.wait()

        sub.start = _start
        sub.stop = _stop
        captured.append(entry)
        return sub

    return _factory


@pytest.mark.asyncio
async def test_a_stream_lost_during_disconnect_leaves_nothing_behind(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    """A sibling pump losing its stream while disconnect is stopping the
    first one must not re-add the socket to _closing (a leak) or close it
    again; afterwards no bookkeeping refers to the socket."""
    captured: list[dict] = []
    release = asyncio.Event()
    with patch(
        "backend.api.conn_manager._Subscription",
        side_effect=_gated_subscription_factory(captured, release),
    ):
        for graph_id in ("graph-1", "graph-2"):
            await connection_manager.subscribe_graph_execs(
                user_id="user-1", graph_id=graph_id, websocket=mock_websocket
            )
        disconnect = asyncio.create_task(
            connection_manager.disconnect_socket(mock_websocket, user_id="user-1")
        )
        while not captured[0].get("stop_started"):
            await asyncio.sleep(0)
        # Mid-teardown, the second subscription loses its stream.
        await captured[1]["on_lost"]("redis_error")
        release.set()
        await asyncio.wait_for(disconnect, timeout=1)

    assert mock_websocket not in connection_manager._closing
    mock_websocket.close.assert_not_awaited()
    assert mock_websocket not in connection_manager._ws_subs
    assert mock_websocket not in connection_manager._ws_notifications
    assert all(
        mock_websocket not in subscribers
        for subscribers in connection_manager.subscriptions.values()
    )


@pytest.mark.asyncio
async def test_concurrent_disconnects_leave_nothing_behind(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    captured: list[dict] = []
    release = asyncio.Event()
    with patch(
        "backend.api.conn_manager._Subscription",
        side_effect=_gated_subscription_factory(captured, release),
    ):
        await connection_manager.subscribe_graph_execs(
            user_id="user-1", graph_id="graph-1", websocket=mock_websocket
        )
        first = asyncio.create_task(
            connection_manager.disconnect_socket(mock_websocket, user_id="user-1")
        )
        while not captured[0].get("stop_started"):
            await asyncio.sleep(0)
        # A second disconnect runs to completion while the first is mid-stop.
        await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")
        release.set()
        await asyncio.wait_for(first, timeout=1)

    assert mock_websocket not in connection_manager._closing
    assert mock_websocket not in connection_manager._ws_subs
    assert connection_manager.subscriptions == {}
