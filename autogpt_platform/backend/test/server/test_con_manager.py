from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket

from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEvent,
    NodeExecutionEvent,
)
from backend.server.conn_manager import ConnectionManager
from backend.server.model import WSMessage, WSMethod


@pytest.fixture
def connection_manager() -> ConnectionManager:
    return ConnectionManager()


@pytest.fixture
def mock_websocket() -> AsyncMock:
    websocket: AsyncMock = AsyncMock(spec=WebSocket)
    websocket.send_text = AsyncMock()
    return websocket


@pytest.mark.asyncio
async def test_connect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    await connection_manager.connect_socket(mock_websocket)
    assert mock_websocket in connection_manager.active_connections
    mock_websocket.accept.assert_called_once()


def test_disconnect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.active_connections.add(mock_websocket)
    connection_manager.subscriptions["test_channel_42"] = {mock_websocket}

    connection_manager.disconnect_socket(mock_websocket)

    assert mock_websocket not in connection_manager.active_connections
    assert mock_websocket not in connection_manager.subscriptions["test_channel_42"]


@pytest.mark.asyncio
async def test_subscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    await connection_manager.subscribe_graph_exec(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        websocket=mock_websocket,
    )
    assert (
        mock_websocket
        in connection_manager.subscriptions["user-1|graph_exec#graph-exec-1"]
    )


@pytest.mark.asyncio
async def test_unsubscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    channel_key = "user-1|graph_exec#graph-exec-1"
    connection_manager.subscriptions[channel_key] = {mock_websocket}

    await connection_manager.unsubscribe(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        websocket=mock_websocket,
    )

    assert "test_graph" not in connection_manager.subscriptions


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
        status=ExecutionStatus.COMPLETED,
        cost=0,
        duration=1.2,
        total_run_time=0.5,
        started_at=datetime.now(tz=timezone.utc),
        ended_at=datetime.now(tz=timezone.utc),
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
