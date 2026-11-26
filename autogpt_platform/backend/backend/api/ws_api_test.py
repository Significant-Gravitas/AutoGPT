import asyncio
import json
import logging
from typing import cast, get_args
from unittest.mock import AsyncMock

import pydantic
import pytest
from annotated_types import MaxLen
from fastapi import WebSocket, WebSocketDisconnect
from pytest_snapshot.plugin import Snapshot
from starlette.websockets import WebSocketState

from backend.api import model as api_model
from backend.api import ws_api
from backend.api.conn_manager import ConnectionManager, _StreamCapacityError
from backend.api.model import WSRequestFailed
from backend.api.test_helpers import override_config
from backend.api.ws_api import (
    AppEnvironment,
    WebsocketServer,
    WSMessage,
    WSMethod,
)
from backend.api.ws_api import app as websocket_app
from backend.api.ws_api import (
    handle_subscribe,
    handle_unsubscribe,
    settings,
    websocket_router,
)
from backend.data.sharded_pubsub import EventStreamLostError
from backend.data.user import DEFAULT_USER_ID
from backend.util import cache_redis
from backend.util.url_redaction import request_credentials

# Only a step that never comes keeps a wait going this long.
_HANG_SECONDS = 10.0


@pytest.fixture(autouse=True)
def account_db(mocker):
    """The closed-account lookup goes through the DatabaseManager, which these
    tests don't run; an account is open unless a test closes it."""
    db = mocker.patch("backend.api.ws_api.user_db").return_value
    db.is_account_closed = AsyncMock(return_value=False)
    # The open socket's per-message closure check, through the same
    # DatabaseManager.
    db.closed = mocker.patch(
        "backend.api.ws_api.is_user_account_closed", new=AsyncMock(return_value=False)
    )
    return db


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


def test_websocket_server_limits_incoming_message_size(mocker) -> None:
    serve_websockets = mocker.patch("backend.api.ws_api.uvicorn.run")

    with override_config(settings, "app_env", AppEnvironment.LOCAL):
        WebsocketServer().run()

    assert serve_websockets.call_args.kwargs["ws_max_size"] <= 16384


def test_websocket_server_keeps_idle_connections_as_long_as_the_api(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_METRICS", "false")
    serve_websockets = mocker.patch("backend.api.ws_api.uvicorn.run")

    with (
        override_config(settings, "app_env", AppEnvironment.LOCAL),
        override_config(settings, "server_keep_alive_timeout_seconds", 777),
    ):
        WebsocketServer().run()

    assert serve_websockets.call_args.kwargs["timeout_keep_alive"] == 777


def test_websocket_server_trusts_x_forwarded_for_from_trusted_proxies(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_METRICS", "false")
    serve_websockets = mocker.patch("backend.api.ws_api.uvicorn.run")

    with (
        override_config(settings, "app_env", AppEnvironment.LOCAL),
        override_config(settings, "trusted_proxies", "172.18.0.0/16"),
    ):
        WebsocketServer().run()

    assert serve_websockets.call_args.kwargs["forwarded_allow_ips"] == "172.18.0.0/16"


def test_every_websocket_request_string_is_length_bounded() -> None:
    """Client-supplied ids become Redis channel names, so each is bounded."""
    requests = [
        model
        for name, model in vars(api_model).items()
        if name.startswith("WS")
        and name.endswith("Request")
        and isinstance(model, type)
        and issubclass(model, pydantic.BaseModel)
    ]
    assert requests, "found no WebSocket request models"
    unbounded = [
        f"{model.__name__}.{field_name}"
        for model in requests
        for field_name, field in model.model_fields.items()
        if str in (field.annotation, *get_args(field.annotation))
        and not any(isinstance(m, MaxLen) for m in field.metadata)
    ]
    assert unbounded == []


def test_an_overlong_id_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError):
        api_model.WSSubscribeGraphExecutionsRequest.model_validate(
            {"graph_id": "g" * (api_model.WS_ID_MAX_LENGTH + 1)}
        )


@pytest.mark.asyncio
async def test_websocket_router_stops_when_the_handshake_is_refused(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mock_manager.connect_socket.return_value = False
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_websocket.receive_text.assert_not_called()
    mock_manager.disconnect_socket.assert_not_called()


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


@pytest.mark.parametrize("enabled", [True, False])
def test_websocket_server_metrics_get_a_listener_of_their_own_only_when_enabled(
    enabled: bool, mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_METRICS", "true" if enabled else "false")
    serve_websockets = mocker.patch("backend.api.ws_api.uvicorn.run")
    serve_metrics = mocker.patch("backend.monitoring.instrumentation.start_http_server")

    WebsocketServer().run()

    config = settings.config
    assert serve_websockets.call_args.kwargs["port"] == config.websocket_server_port
    if not enabled:
        serve_metrics.assert_not_called()
        return
    serve_metrics.assert_called_once_with(
        config.websocket_server_metrics_port, addr=config.pyro_host
    )
    assert config.websocket_server_metrics_port != config.websocket_server_port


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
    reply = WSRequestFailed.model_validate_json(
        mock_websocket.send_text.call_args[0][0]
    )
    assert reply.method == WSMethod.ERROR
    assert reply.success is False
    assert reply.request.model_dump(mode="json") == {
        "method": "graph_execution_event",
        "data": None,
    }
    assert reply.retryable is False
    mock_manager.disconnect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )


async def _reply_to(
    request: str, mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> WSRequestFailed:
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    mock_websocket.receive_text.side_effect = [request, WebSocketDisconnect()]

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_websocket.send_text.assert_called_once()
    return WSRequestFailed.model_validate_json(mock_websocket.send_text.call_args[0][0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, retryable",
    [
        # Redis didn't confirm the SSUBSCRIBE in time.
        (TimeoutError(), True),
        # The shard stream ended, or couldn't be opened.
        (EventStreamLostError("internal detail"), True),
        (ConnectionError("internal detail"), True),
        (_StreamCapacityError("Server busy, try again later"), False),
        (ValueError("Subscription limit reached (1000 per connection)"), False),
        (ValueError("graph_exec #e-1 not found for user #u-1"), False),
    ],
)
async def test_a_failed_subscribe_is_answered_with_itself_and_whether_to_send_it_again(
    failure: Exception,
    retryable: bool,
    mock_websocket: AsyncMock,
    mock_manager: AsyncMock,
    mocker,
) -> None:
    """#16922: the error names the subscription and says whether sending it
    again can succeed, so the client retries what may and drops what can't."""
    mock_manager.subscribe_graph_execs.side_effect = failure
    request = {"method": "subscribe_graph_executions", "data": {"graph_id": "g-1"}}

    reply = await _reply_to(json.dumps(request), mock_websocket, mock_manager, mocker)

    assert reply.method == WSMethod.ERROR
    assert reply.success is False
    assert reply.request.model_dump(mode="json") == request
    assert reply.retryable is retryable
    assert "internal detail" not in (reply.error or "")


@pytest.mark.asyncio
async def test_a_socket_reports_each_kind_of_retryable_failure_once_as_an_error(
    mock_websocket: AsyncMock,
    mock_manager: AsyncMock,
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sentry files error logs, and the client retries these failures: a
    socket's retries of one failure must not each become an event."""
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    request = json.dumps(
        {"method": "subscribe_graph_executions", "data": {"graph_id": "g-1"}}
    )
    mock_websocket.receive_text.side_effect = [request] * 3 + [WebSocketDisconnect()]
    mock_manager.subscribe_graph_execs.side_effect = [
        TimeoutError(),
        TimeoutError(),
        EventStreamLostError("lost"),
    ]

    with caplog.at_level(logging.WARNING, logger="backend.api.ws_api"):
        await websocket_router(
            cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
        )

    levels = [
        record.levelname
        for record in caplog.records
        if record.getMessage().startswith("Error while handling")
    ]
    assert levels == ["ERROR", "WARNING", "ERROR"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [{"graph_id": "g" * (api_model.WS_ID_MAX_LENGTH + 1)}, None],
    ids=["invalid", "missing"],
)
async def test_a_subscribe_with_bad_data_is_not_worth_sending_again(
    data: dict | None, mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    request = {"method": "subscribe_graph_executions", "data": data}

    reply = await _reply_to(json.dumps(request), mock_websocket, mock_manager, mocker)

    assert reply.request.model_dump(mode="json") == request
    assert reply.retryable is False
    mock_manager.subscribe_graph_execs.assert_not_called()


@pytest.mark.asyncio
async def test_websocket_router_sends_nothing_after_the_client_left(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    """A second send on a closed socket raises out of the loop as a traceback."""
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
        WebSocketDisconnect(),
    ]
    mock_manager.subscribe_graph_exec.side_effect = WebSocketDisconnect()

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_websocket.send_text.assert_not_called()
    mock_manager.disconnect_socket.assert_awaited_once()


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
    reply = WSRequestFailed.model_validate_json(
        mock_websocket.send_text.call_args[0][0]
    )
    assert reply.method == WSMethod.ERROR
    assert reply.success is False
    assert reply.request.model_dump(mode="json") == {
        "method": "unsubscribe",
        "data": None,
    }
    assert reply.retryable is False


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


def _auth_frame(token: str = "first-message-token") -> str:
    return json.dumps({"method": "authenticate", "data": {"token": token}})


def _unsigned_socket() -> AsyncMock:
    """A socket with no token in its URL, whose accept moves it to CONNECTED
    as a real one's does."""
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {}

    async def accepted() -> None:
        ws.application_state = WebSocketState.CONNECTED

    ws.accept.side_effect = accepted
    return ws


@pytest.mark.asyncio
async def test_authenticate_websocket_reads_the_token_from_the_first_message(
    mocker,
) -> None:
    from backend.api.ws_api import authenticate_websocket

    parse = mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X", "sid": "session-X"}),
    )
    mocker.patch("backend.api.ws_api.require_active_auth_session", new=AsyncMock())
    ws = _unsigned_socket()
    ws.receive_text.return_value = _auth_frame()

    user_id = await authenticate_websocket(ws)

    assert user_id == "user-X"
    ws.accept.assert_awaited_once()
    ws.close.assert_not_awaited()
    parse.assert_awaited_once_with("first-message-token")
    # Masked in this connection's logs and error reports, as a URL's token is.
    assert "first-message-token" in request_credentials()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        "not json",
        json.dumps({"method": "heartbeat", "data": "ping"}),
        json.dumps({"method": "authenticate", "data": {}}),
        json.dumps({"method": "authenticate", "data": {"token": ""}}),
        json.dumps({"method": "authenticate", "data": "a-token"}),
    ],
)
async def test_authenticate_websocket_without_a_token_message_closes_4001(
    mocker, frame: str
) -> None:
    from backend.api.ws_api import authenticate_websocket

    parse = mocker.patch("backend.api.ws_api.parse_jwt_token_async", new=AsyncMock())
    ws = _unsigned_socket()
    ws.receive_text.return_value = frame

    user_id = await authenticate_websocket(ws)

    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once_with(code=4001, reason="Missing authentication token")
    assert user_id is None
    parse.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_websocket_closes_4001_on_a_binary_first_message(
    mocker,
) -> None:
    from backend.api.ws_api import authenticate_websocket

    ws = _unsigned_socket()
    ws.receive_text.side_effect = KeyError("text")

    assert await authenticate_websocket(ws) is None
    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once_with(code=4001, reason="Missing authentication token")


@pytest.mark.asyncio
async def test_authenticate_websocket_asks_for_a_retry_when_no_message_comes(
    mocker,
) -> None:
    """A tab that paused before it could sign in reconnects; 4001 would stop
    it for good."""
    from backend.api.ws_api import authenticate_websocket

    mocker.patch("backend.api.ws_api.AUTH_MESSAGE_TIMEOUT_S", 0.01)

    async def never() -> str:
        await asyncio.sleep(_HANG_SECONDS)
        return _auth_frame()

    ws = _unsigned_socket()
    ws.receive_text.side_effect = never

    assert (
        await asyncio.wait_for(authenticate_websocket(ws), timeout=_HANG_SECONDS)
        is None
    )
    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once_with(code=1013, reason="Try again later")


@pytest.mark.asyncio
@pytest.mark.parametrize("gone_on", ["accept", "receive"])
async def test_authenticate_websocket_client_gone_before_its_token(
    mocker, gone_on: str
) -> None:
    from backend.api.ws_api import authenticate_websocket

    parse = mocker.patch("backend.api.ws_api.parse_jwt_token_async", new=AsyncMock())
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {}
    if gone_on == "accept":
        ws.accept.side_effect = BrokenPipeError()
    else:
        ws.receive_text.side_effect = WebSocketDisconnect(code=1001)

    assert await authenticate_websocket(ws) is None
    ws.close.assert_not_awaited()
    parse.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_short_first_message_token_is_not_kept_for_masking(mocker) -> None:
    """Masking a few characters would hide ordinary text in later reports."""
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(side_effect=ValueError("bad token")),
    )
    ws = _unsigned_socket()
    ws.receive_text.return_value = _auth_frame(" ")

    assert await authenticate_websocket(ws) is None
    assert " " not in request_credentials()


@pytest.mark.asyncio
async def test_authenticate_websocket_still_reads_a_token_in_the_url(mocker) -> None:
    """Tabs open on an older frontend send it there."""
    from backend.api.ws_api import authenticate_websocket

    parse = mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X", "sid": "session-X"}),
    )
    mocker.patch("backend.api.ws_api.require_active_auth_session", new=AsyncMock())
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "url-token"}

    assert await authenticate_websocket(ws) == "user-X"
    parse.assert_awaited_once_with("url-token")
    # The connection manager accepts it once it has room for the socket.
    ws.accept.assert_not_awaited()
    ws.receive_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refusal_after_the_first_message_does_not_accept_again(
    mocker,
) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(side_effect=ValueError("bad token")),
    )
    ws = _unsigned_socket()
    ws.receive_text.return_value = _auth_frame()

    assert await authenticate_websocket(ws) is None
    ws.accept.assert_awaited_once()
    ws.close.assert_awaited_once_with(code=4003, reason="Invalid token")


@pytest.mark.asyncio
async def test_authenticate_websocket_invalid_token_closes_4003(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(side_effect=ValueError("bad token")),
    )
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4003
    assert user_id is None


@pytest.mark.asyncio
async def test_authenticate_websocket_missing_sub_closes_4002(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"not_sub": "x"}),
    )
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4002
    assert user_id is None


@pytest.mark.asyncio
async def test_authenticate_websocket_happy_path_returns_sub(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X", "sid": "session-X"}),
    )
    validate = mocker.patch(
        "backend.api.ws_api.require_active_auth_session", new=AsyncMock()
    )
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    assert user_id == "user-X"
    validate.assert_awaited_once_with({"sub": "user-X", "sid": "session-X"})


@pytest.mark.asyncio
async def test_authenticate_websocket_closed_account_closes_4003(
    mocker, account_db
) -> None:
    """A token minted before the account closed still verifies for an hour."""
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-closed"}),
    )
    account_db.is_account_closed.return_value = True
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once_with(code=4003, reason="This account is closed.")
    assert user_id is None
    account_db.is_account_closed.assert_awaited_once_with("user-closed")


@pytest.mark.asyncio
async def test_authenticate_websocket_closed_lookup_failure_refuses_with_1011(
    mocker, account_db
) -> None:
    """Closing an account does not ban it, so a closure that can't be read is
    refused, with the outage code rather than the closed one."""
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X"}),
    )
    banned = mocker.patch(
        "backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False)
    )
    account_db.is_account_closed.side_effect = ConnectionError("db manager down")
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once_with(code=1011, reason="Try again")
    assert user_id is None
    banned.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_websocket_refuses_a_banned_account(mocker) -> None:
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X"}),
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=True))
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 4004
    assert user_id is None


@pytest.mark.asyncio
async def test_authenticate_websocket_refuses_when_the_ban_is_unreadable(
    mocker,
) -> None:
    """A new socket fails closed, as a signed-in HTTP route does, and is not
    told it is banned."""
    from backend.api.ws_api import authenticate_websocket

    mocker.patch(
        "backend.api.ws_api.parse_jwt_token_async",
        new=AsyncMock(return_value={"sub": "user-X"}),
    )
    mocker.patch(
        "backend.api.ws_api.is_user_banned",
        new=AsyncMock(side_effect=ConnectionError()),
    )
    ws = AsyncMock(spec=WebSocket)
    ws.query_params = {"token": "abc"}

    user_id = await authenticate_websocket(ws)

    ws.close.assert_awaited_once()
    assert ws.close.call_args.kwargs["code"] == 1011
    assert user_id is None


@pytest.mark.asyncio
async def test_a_ban_ends_an_open_socket_at_its_next_message(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    """The token was checked when the socket opened; a ban set since then
    closes it instead of handling the message."""
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=True))
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
    ]

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    assert mock_websocket.close.call_args.kwargs["code"] == 4004
    mock_manager.subscribe_graph_exec.assert_not_called()
    mock_manager.disconnect_socket.assert_called_once_with(
        mock_websocket, user_id=DEFAULT_USER_ID
    )


@pytest.mark.asyncio
async def test_a_closed_account_ends_an_open_socket_at_its_next_message(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker, account_db
) -> None:
    """Closing an account doesn't ban it, so the per-message check asks about
    the closure too."""
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    account_db.closed.return_value = True
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
    ]

    await websocket_router(
        cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
    )

    mock_websocket.close.assert_awaited_once_with(
        code=4003, reason="This account is closed."
    )
    mock_manager.subscribe_graph_exec.assert_not_called()
    account_db.closed.assert_awaited_once_with(DEFAULT_USER_ID)


@pytest.mark.asyncio
async def test_an_unreadable_closure_keeps_an_open_socket(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker, account_db
) -> None:
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    account_db.closed.side_effect = ConnectionError()
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

    mock_websocket.close.assert_not_awaited()
    mock_manager.subscribe_graph_exec.assert_called_once()


@pytest.mark.asyncio
async def test_an_unreadable_ban_keeps_an_open_socket(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch(
        "backend.api.ws_api.is_user_banned",
        new=AsyncMock(side_effect=ConnectionError()),
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

    mock_websocket.close.assert_not_called()
    mock_manager.subscribe_graph_exec.assert_called_once()


async def _never_answers(user_id: str) -> bool:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


async def _answers_allowed_once_cancelled(user_id: str) -> bool:
    """A layer under the lookup that swallows the cancellation."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ban_lookup",
    [_never_answers, _answers_allowed_once_cancelled],
    ids=["never-answers", "swallows-the-cancel"],
)
async def test_an_account_check_that_runs_out_of_time_closes_the_socket_to_retry(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker, ban_lookup
) -> None:
    """The check never counts as allowed by running out of time: the message
    isn't handled, and the socket closes with a code the client reconnects
    on, which runs the connect-time checks. Nor does a ban lookup that
    swallowed the cancellation start the closure lookup past the deadline,
    which no second cancellation would end."""
    closure_lookup_started = asyncio.Event()

    async def closure_lookup(user_id: str) -> bool:
        closure_lookup_started.set()
        return await _never_answers(user_id)

    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=ban_lookup)
    mocker.patch("backend.api.ws_api.is_user_account_closed", new=closure_lookup)
    # The lookup never finishes on its own, so the bound decides when the
    # check gives up, not whether.
    mocker.patch.object(ws_api, "ACCOUNT_CHECK_TIMEOUT_S", 0.05)
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
        WebSocketDisconnect(),
    ]

    async with asyncio.timeout(_HANG_SECONDS):
        await websocket_router(
            cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
        )

    mock_websocket.close.assert_awaited_once_with(code=1013, reason="Try again")
    mock_manager.subscribe_graph_exec.assert_not_called()
    mock_websocket.send_text.assert_not_called()
    assert not closure_lookup_started.is_set()


@pytest.mark.asyncio
async def test_a_closure_lookup_that_runs_out_of_time_closes_the_socket_to_retry(
    mock_websocket: AsyncMock, mock_manager: AsyncMock, mocker
) -> None:
    """The ban answers "not banned" and the closure lookup never answers: the
    check runs out of time there, and closes the socket with the code the
    client reconnects on, handling nothing."""
    mocker.patch(
        "backend.api.ws_api.authenticate_websocket", return_value=DEFAULT_USER_ID
    )
    mocker.patch("backend.api.ws_api.is_user_banned", new=AsyncMock(return_value=False))
    mocker.patch("backend.api.ws_api.is_user_account_closed", new=_never_answers)
    # The lookup never finishes on its own, so the bound decides when the
    # check gives up, not whether.
    mocker.patch.object(ws_api, "ACCOUNT_CHECK_TIMEOUT_S", 0.05)
    mock_websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC,
            data={"graph_exec_id": "test-graph-exec-1"},
        ).model_dump_json(),
        WebSocketDisconnect(),
    ]

    async with asyncio.timeout(_HANG_SECONDS):
        await websocket_router(
            cast(WebSocket, mock_websocket), cast(ConnectionManager, mock_manager)
        )

    mock_websocket.close.assert_awaited_once_with(code=1013, reason="Try again")
    mock_manager.subscribe_graph_exec.assert_not_called()
    mock_websocket.send_text.assert_not_called()


# The shared cache's own bound, read at import: every test's autouse
# ``patient_shared_cache`` fixture replaces it.
_PRODUCT_REDIS_OP_TIMEOUT_S = cache_redis.REDIS_OP_TIMEOUT_S


def test_the_account_checks_bound_fits_its_two_cache_reads() -> None:
    """The check reads the cache twice, then the database, inside
    ``ACCOUNT_CHECK_TIMEOUT_S``."""
    assert _PRODUCT_REDIS_OP_TIMEOUT_S <= 1.0
    assert 2 * _PRODUCT_REDIS_OP_TIMEOUT_S < ws_api.ACCOUNT_CHECK_TIMEOUT_S


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


# ---------- Health endpoint ----------


def test_health_endpoint_returns_ok() -> None:
    # TestClient triggers lifespan — stub it out so Redis isn't hit.
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
