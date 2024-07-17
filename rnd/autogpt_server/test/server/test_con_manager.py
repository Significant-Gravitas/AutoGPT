from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket

from autogpt_server.data.execution import ExecutionResult, ExecutionStatus
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import Methods, WsMessage


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
    await connection_manager.connect(mock_websocket)
    assert mock_websocket in connection_manager.active_connections
    mock_websocket.accept.assert_called_once()


def test_disconnect(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.active_connections.add(mock_websocket)
    connection_manager.subscriptions["test_graph"] = {mock_websocket}

    connection_manager.disconnect(mock_websocket)

    assert mock_websocket not in connection_manager.active_connections
    assert mock_websocket not in connection_manager.subscriptions["test_graph"]


@pytest.mark.asyncio
async def test_subscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    await connection_manager.subscribe("test_graph", mock_websocket)
    assert mock_websocket in connection_manager.subscriptions["test_graph"]


@pytest.mark.asyncio
async def test_unsubscribe(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.subscriptions["test_graph"] = {mock_websocket}

    await connection_manager.unsubscribe("test_graph", mock_websocket)

    assert "test_graph" not in connection_manager.subscriptions


@pytest.mark.asyncio
async def test_send_execution_result(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    connection_manager.subscriptions["test_graph"] = {mock_websocket}
    result: ExecutionResult = ExecutionResult(
        graph_id="test_graph",
        graph_version=1,
        graph_exec_id="test_exec_id",
        node_exec_id="test_node_exec_id",
        node_id="test_node_id",
        status=ExecutionStatus.COMPLETED,
        input_data={"input1": "value1"},
        output_data={"output1": ["result1"]},
        add_time=datetime.now(tz=timezone.utc),
        queue_time=None,
        start_time=datetime.now(tz=timezone.utc),
        end_time=datetime.now(tz=timezone.utc),
    )

    await connection_manager.send_execution_result(result)

    mock_websocket.send_text.assert_called_once_with(
        WsMessage(
            method=Methods.EXECUTION_EVENT,
            channel="test_graph",
            data=result.model_dump(),
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_send_execution_result_no_subscribers(
    connection_manager: ConnectionManager, mock_websocket: AsyncMock
) -> None:
    result: ExecutionResult = ExecutionResult(
        graph_id="test_graph",
        graph_version=1,
        graph_exec_id="test_exec_id",
        node_exec_id="test_node_exec_id",
        node_id="test_node_id",
        status=ExecutionStatus.COMPLETED,
        input_data={"input1": "value1"},
        output_data={"output1": ["result1"]},
        add_time=datetime.now(),
        queue_time=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    await connection_manager.send_execution_result(result)

    mock_websocket.send_text.assert_not_called()
