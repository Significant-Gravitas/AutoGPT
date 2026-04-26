import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket

from backend.api.conn_manager import (
    ConnectionManager,
    _event_bus_channel,
    _graph_exec_channel_key,
    _graph_execs_channel_key,
    _notification_bus_channel,
    _Subscription,
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
        connection_manager, "_start_notification_subscription", AsyncMock()
    ):
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    assert mock_websocket in connection_manager.active_connections
    mock_websocket.accept.assert_called_once()


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
    full = _event_bus_channel("{u/g}/exec/x")
    # Must start with the configured bus name, then slash, then key.
    assert full.endswith("/{u/g}/exec/x")
    assert "/" in full


def test_notification_bus_channel_prefixes_settings_name() -> None:
    """Notification channels are ``<bus>/<user_id>`` — no wildcard."""
    full = _notification_bus_channel("user-42")
    assert full.endswith("/user-42")
    assert "*" not in full


def test_graph_exec_channel_key_shape() -> None:
    assert _graph_exec_channel_key("u", graph_exec_id="e") == "u|graph_exec#e"


def test_graph_execs_channel_key_shape() -> None:
    assert _graph_execs_channel_key("u", graph_id="g") == "u|graph#g|executions"


# ---------- _Subscription lifecycle ----------


@pytest.mark.asyncio
async def test_subscription_start_issues_ssubscribe_and_primes_channels_map() -> None:
    """start() must SSUBSCRIBE and prime pubsub.channels — redis-py 6.x
    `listen()` exits immediately if `channels` is empty."""
    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}

    async def _listen():
        # Block so the pump doesn't exit immediately.
        await asyncio.sleep(10)
        if False:
            yield

    fake_pubsub.listen = _listen

    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        await sub.start(AsyncMock())
        # SSUBSCRIBE was issued against the full channel.
        fake_pubsub.execute_command.assert_awaited_with("SSUBSCRIBE", "{u/g}/exec/x")
        # channels map primed so listen() doesn't early-exit.
        assert "{u/g}/exec/x" in fake_pubsub.channels
        # stop() must tear everything down cleanly.
        fake_pubsub.aclose = AsyncMock()
        fake_client.aclose = AsyncMock()
        await sub.stop()
        fake_pubsub.aclose.assert_awaited_once()
        fake_client.aclose.assert_awaited_once()


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
    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}
    fake_pubsub.aclose = AsyncMock()

    async def _listen():
        yield {"type": "ssubscribe", "data": 1}
        yield {"type": "smessage", "data": b'{"payload":"one"}'}
        yield {"type": "psubscribe", "data": 1}
        yield {"type": "smessage", "data": b'{"payload":"two"}'}

    fake_pubsub.listen = _listen
    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock()

    received: list = []

    async def on_message(data) -> None:
        received.append(data)

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        await sub.start(on_message)
        await asyncio.sleep(0.05)  # let pump drain
        await sub.stop()

    assert received == [b'{"payload":"one"}', b'{"payload":"two"}']


@pytest.mark.asyncio
async def test_subscription_pump_swallows_handler_exceptions() -> None:
    """A raising handler must not crash the pump — next message still delivers."""
    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}
    fake_pubsub.aclose = AsyncMock()

    async def _listen():
        yield {"type": "smessage", "data": b"boom"}
        yield {"type": "smessage", "data": b"ok"}

    fake_pubsub.listen = _listen
    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock()

    calls: list = []

    async def on_message(data) -> None:
        calls.append(data)
        if data == b"boom":
            raise RuntimeError("handler failed")

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        await sub.start(on_message)
        await asyncio.sleep(0.05)
        await sub.stop()

    assert calls == [b"boom", b"ok"]


@pytest.mark.asyncio
async def test_subscription_pump_logs_and_exits_on_listen_crash() -> None:
    """A `pubsub.listen()` crash must log + exit cleanly, not propagate."""
    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}
    fake_pubsub.aclose = AsyncMock()

    async def _listen():
        raise RuntimeError("listen() exploded")
        yield  # pragma: no cover — unreachable

    fake_pubsub.listen = _listen

    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock()

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        await sub.start(AsyncMock())
        # Let the pump actually crash + log.
        if sub._task is not None:
            await sub._task
        # stop() is still safe to run.
        await sub.stop()


@pytest.mark.asyncio
async def test_subscription_pump_noop_when_pubsub_missing() -> None:
    """Defensive: _pump returns fast if _pubsub was cleared before scheduling."""
    sub = _Subscription("{u/g}/exec/x")
    # Directly invoke _pump with no pubsub — must not raise.
    await sub._pump(AsyncMock())


@pytest.mark.asyncio
async def test_subscription_stop_swallows_sunsubscribe_errors() -> None:
    """A broken cluster must not break graceful teardown."""
    fake_pubsub = MagicMock()

    async def _listen():
        await asyncio.sleep(10)
        if False:
            yield

    fake_pubsub.execute_command = AsyncMock(
        side_effect=[None, RuntimeError("SUNSUB broke")]
    )
    fake_pubsub.channels = {}
    fake_pubsub.listen = _listen
    fake_pubsub.aclose = AsyncMock(side_effect=RuntimeError("close broke"))

    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock(side_effect=RuntimeError("client close broke"))

    sub = _Subscription("{u/g}/exec/x")
    with patch(
        "backend.api.conn_manager.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        await sub.start(AsyncMock())
        # Must not raise even if every teardown call blows up.
        await sub.stop()


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
    """A failed SSUBSCRIBE must not crash connect — the WS stays usable."""
    fake_sub = MagicMock()
    fake_sub.start = AsyncMock(side_effect=RuntimeError("SSUBSCRIBE failed"))
    with patch("backend.api.conn_manager._Subscription", return_value=fake_sub):
        await connection_manager._start_notification_subscription(
            mock_websocket, user_id="user-1"
        )
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
        connection_manager, "_start_notification_subscription", AsyncMock()
    ) as mock_start:
        await connection_manager.connect_socket(mock_websocket, user_id="user-1")
    mock_start.assert_awaited_once_with(mock_websocket, user_id="user-1")
