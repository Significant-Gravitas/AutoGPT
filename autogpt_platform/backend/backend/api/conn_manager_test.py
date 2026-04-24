from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket

from backend.api.conn_manager import ConnectionManager
from backend.api.model import NotificationPayload, WSMessage, WSMethod
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEvent,
    GraphExecutionMeta,
    NodeExecutionEvent,
)


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
    assert mock_websocket in connection_manager.user_connections["user-1"]
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.active_connections.add(mock_websocket)
    connection_manager.subscriptions["test_channel_42"] = {mock_websocket}
    connection_manager.user_connections["user-1"] = {mock_websocket}

    await connection_manager.disconnect_socket(mock_websocket, user_id="user-1")

    assert mock_websocket not in connection_manager.active_connections
    assert "test_channel_42" not in connection_manager.subscriptions
    assert "user-1" not in connection_manager.user_connections


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


@pytest.mark.asyncio
async def test_send_graph_execution_result(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    channel_key = "user-1|graph_exec#graph-exec-1"
    connection_manager.subscriptions[channel_key] = {mock_websocket}
    result = GraphExecutionEvent(
        id="graph-exec-1",
        user_id="user-1",
        graph_id="test_graph",
        graph_version=1,
        preset_id=None,
        status=ExecutionStatus.COMPLETED,
        started_at=datetime.now(tz=timezone.utc),
        ended_at=datetime.now(tz=timezone.utc),
        stats=GraphExecutionEvent.Stats(
            cost=0,
            duration=1.2,
            node_exec_time=0.5,
            node_exec_count=2,
        ),
        inputs={
            "input_1": "some input value :)",
            "input_2": "some *other* input value",
        },
        credential_inputs=None,
        nodes_input_masks=None,
        outputs={
            "the_output": ["some output value"],
            "other_output": ["sike there was another output"],
        },
    )

    await connection_manager.send_execution_update(result)

    mock_websocket.send_text.assert_called_once_with(
        WSMessage(
            method=WSMethod.GRAPH_EXECUTION_EVENT,
            channel="user-1|graph_exec#graph-exec-1",
            data=result.model_dump(),
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_send_node_execution_result(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    channel_key = "user-1|graph_exec#graph-exec-1"
    connection_manager.subscriptions[channel_key] = {mock_websocket}
    result = NodeExecutionEvent(
        user_id="user-1",
        graph_id="test_graph",
        graph_version=1,
        graph_exec_id="graph-exec-1",
        node_exec_id="test_node_exec_id",
        node_id="test_node_id",
        block_id="test_block_id",
        status=ExecutionStatus.COMPLETED,
        input_data={"input1": "value1"},
        output_data={"output1": ["result1"]},
        add_time=datetime.now(tz=timezone.utc),
        queue_time=None,
        start_time=datetime.now(tz=timezone.utc),
        end_time=datetime.now(tz=timezone.utc),
    )

    await connection_manager.send_execution_update(result)

    mock_websocket.send_text.assert_called_once_with(
        WSMessage(
            method=WSMethod.NODE_EXECUTION_EVENT,
            channel="user-1|graph_exec#graph-exec-1",
            data=result.model_dump(),
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_send_execution_result_user_mismatch(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    channel_key = "user-1|graph_exec#graph-exec-1"
    connection_manager.subscriptions[channel_key] = {mock_websocket}
    result = NodeExecutionEvent(
        user_id="user-2",
        graph_id="test_graph",
        graph_version=1,
        graph_exec_id="graph-exec-1",
        node_exec_id="test_node_exec_id",
        node_id="test_node_id",
        block_id="test_block_id",
        status=ExecutionStatus.COMPLETED,
        input_data={"input1": "value1"},
        output_data={"output1": ["result1"]},
        add_time=datetime.now(tz=timezone.utc),
        queue_time=None,
        start_time=datetime.now(tz=timezone.utc),
        end_time=datetime.now(tz=timezone.utc),
    )

    await connection_manager.send_execution_update(result)

    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_send_execution_result_no_subscribers(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    result = NodeExecutionEvent(
        user_id="user-1",
        graph_id="test_graph",
        graph_version=1,
        graph_exec_id="test_exec_id",
        node_exec_id="test_node_exec_id",
        node_id="test_node_id",
        block_id="test_block_id",
        status=ExecutionStatus.COMPLETED,
        input_data={"input1": "value1"},
        output_data={"output1": ["result1"]},
        add_time=datetime.now(),
        queue_time=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    await connection_manager.send_execution_update(result)

    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_send_notification(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.user_connections["user-1"] = {mock_websocket}

    await connection_manager.send_notification(
        user_id="user-1", payload=NotificationPayload(type="info", event="hey")
    )

    mock_websocket.send_text.assert_called_once()
    sent_message = mock_websocket.send_text.call_args[0][0]
    expected_message = WSMessage(
        method=WSMethod.NOTIFICATION,
        data={"type": "info", "event": "hey"},
    ).model_dump_json()
    assert sent_message == expected_message
