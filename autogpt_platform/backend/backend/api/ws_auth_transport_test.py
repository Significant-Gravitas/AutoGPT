import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
import uvicorn
from fastapi import HTTPException, WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from backend.api import ws_api
from backend.api.conn_manager import ConnectionManager


@pytest_asyncio.fixture
async def websocket_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[str, ConnectionManager]]:
    manager = ConnectionManager()
    monkeypatch.setattr(manager, "_start_notification_subscription", AsyncMock())
    monkeypatch.setattr(ws_api, "_connection_manager", manager)
    monkeypatch.setattr(ws_api.settings.config, "enable_auth", True)
    server = uvicorn.Server(
        uvicorn.Config(
            ws_api.app,
            host="127.0.0.1",
            port=0,
            lifespan="off",
            ws="websockets-sansio",
            log_config=None,
        )
    )
    serving = asyncio.create_task(server.serve())

    async def wait_until_ready() -> None:
        while not server.started:
            if serving.done():
                await serving
                pytest.fail("Uvicorn exited before readiness")
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_until_ready(), timeout=10)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws", manager
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=5)


@pytest.mark.parametrize(
    "failure,code,reason",
    [
        ("unavailable", 1013, "Authentication service unavailable"),
        ("revoked", 4003, "Session expired or revoked"),
        ("invalid-token", 4003, "Invalid token"),
        ("missing-sub", 4002, "Invalid token"),
        ("missing-token", 4001, "Missing authentication token"),
    ],
)
async def test_auth_failure_reaches_the_client_as_a_close_frame(
    websocket_server: tuple[str, ConnectionManager],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    code: int,
    reason: str,
) -> None:
    url, manager = websocket_server
    parse = AsyncMock(return_value={} if failure == "missing-sub" else {"sub": "user"})
    if failure == "invalid-token":
        parse.side_effect = ValueError("invalid")
    validate = AsyncMock()
    if failure in {"unavailable", "revoked"}:
        validate.side_effect = HTTPException(503 if failure == "unavailable" else 401)
    monkeypatch.setattr(ws_api, "parse_jwt_token_async", parse)
    monkeypatch.setattr(ws_api, "require_active_auth_session", validate)
    track_connection = Mock()
    monkeypatch.setattr(ws_api, "update_websocket_connections", track_connection)

    target = url if failure == "missing-token" else f"{url}?token=wire-test"
    async with connect(target, open_timeout=3, close_timeout=1) as websocket:
        with pytest.raises(ConnectionClosedError) as closed:
            await asyncio.wait_for(websocket.recv(), timeout=3)
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == code
        assert closed.value.rcvd.reason == reason

    assert not manager.active_connections
    assert not manager._ws_subs
    assert not manager._ws_notifications
    track_connection.assert_not_called()


@pytest.mark.parametrize("disconnect_on", ["accept", "close"])
async def test_disconnected_auth_failure_does_not_register_socket(
    monkeypatch: pytest.MonkeyPatch, disconnect_on: str
) -> None:
    monkeypatch.setattr(ws_api.settings.config, "enable_auth", True)
    send = AsyncMock(
        side_effect=(
            [BrokenPipeError()]
            if disconnect_on == "accept"
            else [None, BrokenPipeError()]
        )
    )
    websocket = WebSocket(
        {"type": "websocket", "query_string": b""},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=send,
    )
    manager = ConnectionManager()
    connect_socket = AsyncMock()
    monkeypatch.setattr(manager, "connect_socket", connect_socket)
    track_connection = Mock()
    monkeypatch.setattr(ws_api, "update_websocket_connections", track_connection)

    await ws_api.websocket_router(websocket, manager)

    assert send.await_count == (1 if disconnect_on == "accept" else 2)
    connect_socket.assert_not_awaited()
    assert not manager.active_connections
    assert not manager._ws_subs
    assert not manager._ws_notifications
    track_connection.assert_not_called()


@pytest.mark.parametrize("fail_on", ["accept", "close"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_auth_failure_propagates_unrelated_errors(
    fail_on: str, error_type: type[BaseException]
) -> None:
    failure = error_type("send failed")
    websocket = WebSocket(
        {"type": "websocket"},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=AsyncMock(
            side_effect=[failure] if fail_on == "accept" else [None, failure]
        ),
    )

    with pytest.raises(error_type, match="send failed"):
        await ws_api._close_authentication_failure(
            websocket, code=4001, reason="Missing authentication token"
        )


@pytest.mark.parametrize("enable_auth", [True, False])
async def test_valid_socket_is_accepted_once_and_serves_heartbeats(
    websocket_server: tuple[str, ConnectionManager],
    monkeypatch: pytest.MonkeyPatch,
    enable_auth: bool,
) -> None:
    url, manager = websocket_server
    monkeypatch.setattr(ws_api.settings.config, "enable_auth", enable_auth)
    monkeypatch.setattr(
        ws_api, "parse_jwt_token_async", AsyncMock(return_value={"sub": "user"})
    )
    monkeypatch.setattr(ws_api, "require_active_auth_session", AsyncMock())
    async with connect(f"{url}?token=wire-test", open_timeout=3) as websocket:
        await websocket.send(json.dumps({"method": "heartbeat"}))
        response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3))
        assert response == {"method": "heartbeat", "data": "pong", "success": True}
        assert len(manager.active_connections) == 1

    async def wait_until_disconnected() -> None:
        while manager.active_connections:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_disconnected(), timeout=3)
