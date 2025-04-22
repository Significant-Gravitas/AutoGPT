from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from backend.data.user import DEFAULT_USER_ID
from backend.server.conn_manager import ConnectionManager
from backend.server.ws_api import (
    WSMessage,
    WSMethod,
    handle_subscribe,
    handle_unsubscribe,
    websocket_router,
)


@pytest.fixture
def mock_websocket() -> AsyncMock:
    return AsyncMock(spec=WebSocket)


@pytest.fixture
def mock_manager() -> AsyncMock:
    return AsyncMock(spec=ConnectionManager)


@pytest.mark.asyncio
async def test_websocket_router_subscribe(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
        WebSocketDisconnect(),
    ]
    mock_manager.subscribe_graph_exec.return_value = (
        f"{DEFAULT_USER_ID}|graph_exec#test-graph-exec-1"
    )

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_manager.connect_socket.assert_called_once_with(mock_websocket)
    mock_manager.subscribe_graph_exec.assert_called_once_with(
        user_id=DEFAULT_USER_ID,
        graph_exec_id="test-graph-exec-1",
        websocket=mock_websocket,
    )
    mock_websocket.send_text.assert_called_once()
    assert (
        '"method":"subscribe_graph_execution"'
        in mock_websocket.send_text.call_args[0][0]
    )
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]
    mock_manager.disconnect_socket.assert_called_once_with(mock_websocket)


@pytest.mark.asyncio
async def test_websocket_router_unsubscribe(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.UNSUBSCRIBE,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
        WebSocketDisconnect(),
    ]
    mock_manager.unsubscribe_graph_exec.return_value = (
        f"{DEFAULT_USER_ID}|graph_exec#test-graph-exec-1"
    )

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_manager.connect_socket.assert_called_once_with(mock_websocket)
    mock_manager.unsubscribe_graph_exec.assert_called_once_with(
        user_id=DEFAULT_USER_ID,
        graph_exec_id="test-graph-exec-1",
        websocket=mock_websocket,
    )
    mock_websocket.send_text.assert_called_once()
    assert '"method":"unsubscribe"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]
    mock_manager.disconnect_socket.assert_called_once_with(mock_websocket)


@pytest.mark.asyncio
async def test_websocket_router_invalid_method(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    mock_websocket.receive_text.side_effect = [
        WSMessage(method=WSMethod.GRAPH_EXECUTION_EVENT).model_dump_json(),
        WebSocketDisconnect(),
    ]

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_manager.connect_socket.assert_called_once_with(mock_websocket)
    mock_websocket.send_text.assert_called_once()
    assert '"method":"error"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":false' in mock_websocket.send_text.call_args[0][0]
    mock_manager.disconnect_socket.assert_called_once_with(mock_websocket)


@pytest.mark.asyncio
async def test_handle_subscribe_success(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    message = WSMessage(
        method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
        data={"graph_exec_id": "test-graph-exec-id"},
    )
    mock_manager.subscribe_graph_exec.return_value = (
        "user-1|graph_exec#test-graph-exec-id"
    )

    await handle_subscribe(
        connection_manager=cast(ConnectionManager, mock_manager),
        websocket=cast(WebSocket, mock_websocket),
        user_id="user-1",
        message=message,
    )

    mock_manager.subscribe_graph_exec.assert_called_once_with(
        user_id="user-1",
        graph_exec_id="test-graph-exec-id",
        websocket=mock_websocket,
    )
    mock_websocket.send_text.assert_called_once()
    assert (
        '"method":"subscribe_graph_execution"'
        in mock_websocket.send_text.call_args[0][0]
    )
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_subscribe_missing_data(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    message = WSMessage(method=WSMethod.SUBSCRIBE_GRAPH_EXEC)

    await handle_subscribe(
        connection_manager=cast(ConnectionManager, mock_manager),
        websocket=cast(WebSocket, mock_websocket),
        user_id="user-1",
        message=message,
    )

    mock_manager.subscribe_graph_exec.assert_not_called()
    mock_websocket.send_text.assert_called_once()
    assert '"method":"error"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":false' in mock_websocket.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_unsubscribe_success(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    message = WSMessage(
        method=WSMethod.UNSUBSCRIBE, data={"graph_exec_id": "test-graph-exec-id"}
    )
    mock_manager.unsubscribe_graph_exec.return_value = (
        "user-1|graph_exec#test-graph-exec-id"
    )

    await handle_unsubscribe(
        connection_manager=cast(ConnectionManager, mock_manager),
        websocket=cast(WebSocket, mock_websocket),
        user_id="user-1",
        message=message,
    )

    mock_manager.unsubscribe_graph_exec.assert_called_once_with(
        user_id="user-1",
        graph_exec_id="test-graph-exec-id",
        websocket=mock_websocket,
    )
    mock_websocket.send_text.assert_called_once()
    assert '"method":"unsubscribe"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_unsubscribe_missing_data(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    message = WSMessage(method=WSMethod.UNSUBSCRIBE)

    await handle_unsubscribe(
        connection_manager=cast(ConnectionManager, mock_manager),
        websocket=cast(WebSocket, mock_websocket),
        user_id="user-1",
        message=message,
    )

    mock_manager._unsubscribe.assert_not_called()
    mock_websocket.send_text.assert_called_once()
    assert '"method":"error"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":false' in mock_websocket.send_text.call_args[0][0]
