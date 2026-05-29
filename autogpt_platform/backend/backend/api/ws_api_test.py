import json
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from pytest_snapshot.plugin import Snapshot

from backend.api.conn_manager import ConnectionManager
from backend.api.test_helpers import override_config
from backend.api.ws_api import AppEnvironment, WebsocketServer, WSMessage, WSMethod
from backend.api.ws_api import app as websocket_app
from backend.api.ws_api import (
    handle_subscribe,
    handle_unsubscribe,
    settings,
    websocket_router,
)
from backend.data.user import DEFAULT_USER_ID


@pytest.fixture
def mock_websocket() -> AsyncMock:
    mock = AsyncMock(spec=WebSocket)
    mock.query_params = {}  # Add query_params attribute for authentication
    return mock


@pytest.fixture
def mock_manager() -> AsyncMock:
    return AsyncMock(spec=ConnectionManager)


def test_websocket_server_uses_cors_helper(mocker) -> None:
    cors_params = {
        "allow_origins": ["https://app.example.com"],
        "allow_origin_regex": None,
    }
    mocker.patch("backend.api.ws_api.uvicorn.run")
    cors_middleware = mocker.patch(
        "backend.api.ws_api.CORSMiddleware", return_value=object()
    )
    build_cors = mocker.patch(
        "backend.api.ws_api.build_cors_params", return_value=cors_params
    )

    with (
        override_config(
            settings, "backend_cors_allow_origins", cors_params["allow_origins"]
        ),
        override_config(settings, "app_env", AppEnvironment.LOCAL),
    ):
        WebsocketServer().run()

    build_cors.assert_called_once_with(
        cors_params["allow_origins"], AppEnvironment.LOCAL
    )
    cors_middleware.assert_called_once_with(
        app=websocket_app,
        allow_origins=cors_params["allow_origins"],
        allow_origin_regex=cors_params["allow_origin_regex"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def test_websocket_server_blocks_localhost_in_production(mocker) -> None:
    mocker.patch("backend.api.ws_api.uvicorn.run")

    with (
        override_config(
            settings, "backend_cors_allow_origins", ["http://localhost:3000"]
        ),
        override_config(settings, "app_env", AppEnvironment.PRODUCTION),
    ):
        with pytest.raises(ValueError):
            WebsocketServer().run()


@pytest.mark.asyncio
async def test_websocket_router_subscribe(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, snapshot: Snapshot, mocker
) -> None:
    # Mock the authenticate_websocket function to ensure it returns a valid user_id
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )

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

    mock_manager.connect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )
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

    # Capture and snapshot the WebSocket response message
    sent_message = mock_websocket.send_text.call_args[0][0]
    parsed_message = json.loads(sent_message)
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(parsed_message, indent=2, sort_keys=True), "sub")

    mock_manager.disconnect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )


@pytest.mark.asyncio
async def test_websocket_router_unsubscribe(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, snapshot: Snapshot, mocker
) -> None:
    # Mock the authenticate_websocket function to ensure it returns a valid user_id
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )

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

    mock_manager.connect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )
    mock_manager.unsubscribe_graph_exec.assert_called_once_with(
        user_id=DEFAULT_USER_ID,
        graph_exec_id="test-graph-exec-1",
        websocket=mock_websocket,
    )
    mock_websocket.send_text.assert_called_once()
    assert '"method":"unsubscribe"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]

    # Capture and snapshot the WebSocket response message
    sent_message = mock_websocket.send_text.call_args[0][0]
    parsed_message = json.loads(sent_message)
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(parsed_message, indent=2, sort_keys=True), "unsub")

    mock_manager.disconnect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )


@pytest.mark.asyncio
async def test_websocket_router_invalid_method(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    # Mock the authenticate_websocket function to ensure it returns a valid user_id
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )

    mock_websocket.receive_text.side_effect = [
        WSMessage(method=WSMethod.GRAPH_EXECUTION_EVENT).model_dump_json(),
        WebSocketDisconnect(),
    ]

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_manager.connect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )
    mock_websocket.send_text.assert_called_once()
    assert '"method":"error"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":false' in mock_websocket.send_text.call_args[0][0]
    mock_manager.disconnect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )


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

    mock_manager.unsubscribe_graph_exec.assert_not_called()
    mock_websocket.send_text.assert_called_once()
    assert '"method":"error"' in mock_websocket.send_text.call_args[0][0]
    assert '"success":false' in mock_websocket.send_text.call_args[0][0]


# ---------- Per-graph subscribe branch ----------


@pytest.mark.asyncio
async def test_handle_subscribe_graph_execs_branch(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    """The SUBSCRIBE_GRAPH_EXECS branch must route to subscribe_graph_execs,
    not subscribe_graph_exec — regression guard for the aggregate channel."""
    message = WSMessage(
        method=WSMethod.SUBSCRIBE_GRAPH_EXECS,
        data={"graph_id": "graph-abc"},
    )
    mock_manager.subscribe_graph_execs.return_value = (
        "user-1|graph#graph-abc|executions"
    )

    await handle_subscribe(
        connection_manager=cast(ConnectionManager, mock_manager),
        websocket=cast(WebSocket, mock_websocket),
        user_id="user-1",
        message=message,
    )

    mock_manager.subscribe_graph_execs.assert_called_once_with(
        user_id="user-1",
        graph_id="graph-abc",
        websocket=mock_websocket,
    )
    mock_manager.subscribe_graph_exec.assert_not_called()
    mock_websocket.send_text.assert_called_once()
    assert (
        '"method":"subscribe_graph_executions"'
        in mock_websocket.send_text.call_args[0][0]
    )
    assert '"success":true' in mock_websocket.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_handle_subscribe_rejects_unrelated_method(
    mock_websocket: AsyncMock, mock_manager: AsyncMock
) -> None:
    """handle_subscribe must raise for methods that aren't SUBSCRIBE_*."""
    import pytest as _pytest

    message = WSMessage(
        method=WSMethod.HEARTBEAT,
        data={"graph_exec_id": "x"},
    )

    with _pytest.raises(ValueError):
        await handle_subscribe(
            connection_manager=cast(ConnectionManager, mock_manager),
            websocket=cast(WebSocket, mock_websocket),
            user_id="user-1",
            message=message,
        )


# ---------- authenticate_websocket branches ----------


@pytest.mark.asyncio
async def test_authenticate_websocket_missing_token_closes_4001(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch.object(settings.config, "enable_auth", True)
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4001
    assert user_id == ""


@pytest.mark.asyncio
async def test_authenticate_websocket_invalid_token_closes_4003(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch.object(settings.config, "enable_auth", True)
    mocker.patch(
        "backend.api.ws_api.parse_jwt_token", side_effect=ValueError("bad token")
    )
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4003
    assert user_id == ""


@pytest.mark.asyncio
async def test_authenticate_websocket_missing_sub_closes_4002(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch.object(settings.config, "enable_auth", True)
    mocker.patch("backend.api.ws_api.parse_jwt_token", return_value={"not_sub": "x"})
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4002
    assert user_id == ""


@pytest.mark.asyncio
async def test_authenticate_websocket_happy_path_returns_sub(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch.object(settings.config, "enable_auth", True)
    mocker.patch("backend.api.ws_api.parse_jwt_token", return_value={"sub": "user-X"})
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    assert user_id == "user-X"


@pytest.mark.asyncio
async def test_authenticate_websocket_auth_disabled_returns_default(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch.object(settings.config, "enable_auth", False)
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {}

    user_id = await authenticate_websocket(ws)

    assert user_id == DEFAULT_USER_ID


# ---------- get_connection_manager singleton ----------


def test_get_connection_manager_singleton() -> None:
    """Repeated calls must return the same ConnectionManager — the WS router
    depends on a single process-wide subscription table."""
    import backend.api.ws_api as ws_api

    ws_api._connection_manager = None
    a = ws_api.get_connection_manager()
    b = ws_api.get_connection_manager()
    assert a is b
    assert isinstance(a, ConnectionManager)


# ---------- Lifespan: Prisma connect/disconnect ----------


@pytest.mark.asyncio
async def test_lifespan_connects_and_disconnects_prisma(mocker) -> None:
    """Lifespan must both connect() and disconnect() db — the subscribe path
    resolves graph_id via Prisma so a missing connect() is the regression bug."""
    from fastapi import FastAPI

    from backend.api.ws_api import lifespan

    mock_db = mocker.patch("backend.api.ws_api.db")
    mock_db.connect = AsyncMock()
    mock_db.disconnect = AsyncMock()

    dummy_app = FastAPI()
    async with lifespan(dummy_app):
        mock_db.connect.assert_awaited_once()
        mock_db.disconnect.assert_not_called()
    mock_db.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_still_disconnects_on_exception(mocker) -> None:
    """If the app raises inside the yield, Prisma must still disconnect."""
    from fastapi import FastAPI

    from backend.api.ws_api import lifespan

    mock_db = mocker.patch("backend.api.ws_api.db")
    mock_db.connect = AsyncMock()
    mock_db.disconnect = AsyncMock()

    dummy_app = FastAPI()

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with lifespan(dummy_app):
            raise _Boom()

    mock_db.disconnect.assert_awaited_once()


# ---------- Health endpoint ----------


def test_health_endpoint_returns_ok() -> None:
    # TestClient triggers lifespan — stub it out so Prisma isn't hit.
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    import backend.api.ws_api as ws_api

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    # Replace the app-level lifespan temporarily.
    real_router_lifespan = ws_api.app.router.lifespan_context
    ws_api.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(ws_api.app) as client:
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}
    finally:
        ws_api.app.router.lifespan_context = real_router_lifespan
