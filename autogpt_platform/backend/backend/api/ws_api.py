import logging
from contextlib import asynccontextmanager
from typing import Protocol

import pydantic
import uvicorn
from autogpt_libs.auth.closed_accounts import ACCOUNT_CLOSED_DETAIL
from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from autogpt_libs.auth.jwt_utils import parse_jwt_token_async
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from backend.api.conn_manager import ConnectionManager, _is_ws_close_race
from backend.api.model import (
    WSMessage,
    WSMethod,
    WSRequestFailed,
    WSRequestSent,
    WSSubscribeGraphExecutionRequest,
    WSSubscribeGraphExecutionsRequest,
)
from backend.api.session_auth import (
    SESSION_TOKEN_RENEWAL_REQUIRED,
    require_active_auth_session,
)
from backend.api.utils.cors import build_cors_params
from backend.api.utils.validation_errors import loggable_validation_errors
from backend.data import redis_client
from backend.data.db_accessors import auth_session_db, user_db
from backend.data.user import is_user_banned, is_user_refused
from backend.monitoring.instrumentation import (
    instrument_fastapi,
    start_metrics_server,
    update_websocket_connections,
)
from backend.util.request_credentials import remember_request_credentials
from backend.util.service import AppProcess
from backend.util.settings import AppEnvironment, Config, Settings
from backend.util.trusted_proxies import uvicorn_forwarded_allow_ips

logger = logging.getLogger(__name__)
settings = Settings()

ACCOUNT_BANNED_CLOSE_CODE = 4004
# Clients send only subscribe, unsubscribe and heartbeat messages of ~100 bytes.
WS_MAX_MESSAGE_BYTES = 4096


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No Prisma: reads go through the DatabaseManager (db_accessors), so this
    # server needs no database URL. Eager Redis connect to fail fast.
    await redis_client.get_redis_async()
    try:
        # Cold RPC-client imports must finish before request auth deadlines start.
        auth_session_db()
        yield
    finally:
        # Silences asyncio's "Unclosed ClusterNode" GC warning at shutdown.
        try:
            await redis_client.disconnect_async()
        except Exception:
            logger.warning("redis_client.disconnect_async failed", exc_info=True)


docs_url = "/docs" if settings.config.app_env == AppEnvironment.LOCAL else None
app = FastAPI(lifespan=lifespan, docs_url=docs_url)
# Keeps each connection's credentials for the error reports raised handling it.
remember_request_credentials(app)
_connection_manager = None

# Add Prometheus instrumentation. The metrics are served by WebsocketServer.run
# on their own port, never on this public app.
instrument_fastapi(app, service_name="websocket-server", expose_endpoint=False)


def get_connection_manager():
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def authenticate_websocket(websocket: WebSocket) -> str:
    token = websocket.query_params.get("token")
    if not token:
        await _close_authentication_failure(
            websocket, code=4001, reason="Missing authentication token"
        )
        return ""
    try:
        payload = await parse_jwt_token_async(token)
        user_id = payload.get("sub")
        if not user_id:
            await _close_authentication_failure(
                websocket, code=4002, reason="Invalid token"
            )
            return ""
        if not await _account_can_connect(websocket, user_id):
            return ""
        await require_active_auth_session(payload)
        return user_id
    except ValueError:
        await _close_authentication_failure(
            websocket, code=4003, reason="Invalid token"
        )
        return ""
    except HTTPException as exc:
        if (exc.headers or {}).get("X-Error-Code") == SESSION_TOKEN_RENEWAL_REQUIRED:
            await _close_authentication_failure(
                websocket, code=1013, reason="Session token requires renewal"
            )
        elif exc.status_code == 503:
            await _close_authentication_failure(
                websocket, code=1013, reason="Authentication service unavailable"
            )
        elif exc.status_code == 403 and exc.detail == ACCOUNT_BANNED_DETAIL:
            await _close_authentication_failure(
                websocket, code=ACCOUNT_BANNED_CLOSE_CODE, reason="Banned"
            )
        else:
            await _close_authentication_failure(
                websocket, code=4003, reason="Session expired or revoked"
            )
        return ""


async def _account_can_connect(websocket: WebSocket, user_id: str) -> bool:
    # A token minted before the account closed still verifies. Refused when
    # the closure can't be read, as the ban below is: closing an account does
    # not ban it. 1011, so an outage never reads as a closed account.
    try:
        closed = await user_db().is_account_closed(user_id)
    except Exception:
        logger.warning(
            f"Closed-account lookup failed for user #{user_id}; refusing the socket"
        )
        await _close_authentication_failure(websocket, code=1011, reason="Try again")
        return False
    if closed:
        await _close_authentication_failure(
            websocket, code=4003, reason=ACCOUNT_CLOSED_DETAIL
        )
        return False

    # Refused when the ban can't be read, as a request to a signed-in route is;
    # 1011 rather than the ban's code, so an outage never reads as a ban.
    try:
        banned = await is_user_banned(user_id)
    except Exception:
        logger.warning(f"Ban lookup failed for user #{user_id}; refusing the socket")
        await _close_authentication_failure(websocket, code=1011, reason="Try again")
        return False
    if banned:
        await _close_authentication_failure(
            websocket, code=ACCOUNT_BANNED_CLOSE_CODE, reason="Banned"
        )
        return False
    return True


async def _close_authentication_failure(
    websocket: WebSocket, *, code: int, reason: str
) -> None:
    """Complete the handshake so the client receives the close frame."""
    try:
        await websocket.accept()
        await websocket.close(code=code, reason=reason)
    except (WebSocketDisconnect, OSError):
        logger.debug("WebSocket client disconnected during authentication rejection")


# ===================== Message Handlers ===================== #


class WSMessageHandler(Protocol):
    async def __call__(
        self,
        connection_manager: ConnectionManager,
        websocket: WebSocket,
        user_id: str,
        message: WSMessage,
    ): ...


async def handle_subscribe(
    connection_manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WSMessage,
):
    if not message.data:
        await _answer_failure(
            websocket, message, "Subscription data missing", retryable=False
        )
        return

    # Verify that user has read access to graph
    # if not get_db_client().get_graph(
    #     graph_id=sub_req.graph_id,
    #     version=sub_req.graph_version,
    #     user_id=user_id,
    # ):
    #     await websocket.send_text(
    #         WsMessage(
    #             method=Methods.ERROR,
    #             success=False,
    #             error="Access denied",
    #         ).model_dump_json()
    #     )
    #     return

    if message.method == WSMethod.SUBSCRIBE_GRAPH_EXEC:
        sub_req = WSSubscribeGraphExecutionRequest.model_validate(message.data)
        channel_key = await connection_manager.subscribe_graph_exec(
            user_id=user_id,
            graph_exec_id=sub_req.graph_exec_id,
            websocket=websocket,
        )

    elif message.method == WSMethod.SUBSCRIBE_GRAPH_EXECS:
        sub_req = WSSubscribeGraphExecutionsRequest.model_validate(message.data)
        channel_key = await connection_manager.subscribe_graph_execs(
            user_id=user_id,
            graph_id=sub_req.graph_id,
            websocket=websocket,
        )

    else:
        raise ValueError(
            f"{handle_subscribe.__name__} can't handle '{message.method}' messages"
        )

    logger.debug(f"New subscription on channel {channel_key} for user #{user_id}")
    await websocket.send_text(
        WSMessage(
            method=message.method,
            success=True,
            channel=channel_key,
        ).model_dump_json()
    )


async def handle_unsubscribe(
    connection_manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WSMessage,
):
    if not message.data:
        await _answer_failure(
            websocket, message, "Subscription data missing", retryable=False
        )
        return

    unsub_req = WSSubscribeGraphExecutionRequest.model_validate(message.data)
    channel_key = await connection_manager.unsubscribe_graph_exec(
        user_id=user_id,
        graph_exec_id=unsub_req.graph_exec_id,
        websocket=websocket,
    )

    logger.debug(f"Removed subscription on channel {channel_key} for user #{user_id}")
    await websocket.send_text(
        WSMessage(
            method=WSMethod.UNSUBSCRIBE,
            success=True,
            channel=channel_key,
        ).model_dump_json()
    )


async def handle_heartbeat(
    connection_manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WSMessage,
):
    await websocket.send_json(
        {
            "method": WSMethod.HEARTBEAT.value,
            "data": "pong",
            "success": True,
        }
    )


_MSG_HANDLERS: dict[WSMethod, WSMessageHandler] = {
    WSMethod.HEARTBEAT: handle_heartbeat,
    WSMethod.SUBSCRIBE_GRAPH_EXEC: handle_subscribe,
    WSMethod.SUBSCRIBE_GRAPH_EXECS: handle_subscribe,
    WSMethod.UNSUBSCRIBE: handle_unsubscribe,
}


# ===================== WebSocket Server ===================== #


@app.websocket("/ws")
async def websocket_router(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_connection_manager)
):
    user_id = await authenticate_websocket(websocket)
    if not user_id:
        return
    if not await manager.connect_socket(websocket, user_id=user_id):
        return

    # Track WebSocket connection
    update_websocket_connections(user_id, 1)
    # Each kind of failure the client may retry, once this socket has hit it.
    failed_before: set[type[Exception]] = set()

    try:
        while True:
            data = await websocket.receive_text()
            # The socket outlives its token, so a ban or a closed account ends
            # it at the next message. A ban was ruled out first, so a refusal
            # here is a closed account.
            if await _banned_now(user_id):
                await websocket.close(code=ACCOUNT_BANNED_CLOSE_CODE, reason="Banned")
                break
            if await _refused_now(user_id):
                await websocket.close(code=4003, reason=ACCOUNT_CLOSED_DETAIL)
                break
            try:
                message = WSMessage.model_validate_json(data)
            except pydantic.ValidationError as e:
                logger.error(
                    "Invalid WebSocket message from user #%s: %s",
                    user_id,
                    loggable_validation_errors(e),
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error=("Invalid message format. Review the schema and retry"),
                    ).model_dump_json()
                )
                continue

            if message.method in _MSG_HANDLERS:
                handled = await _handle_request(
                    manager, websocket, user_id, message, failed_before
                )
                if not handled:
                    break
                continue

            if message.method == WSMethod.ERROR:
                logger.error(f"WebSocket Error message received: {message.data}")

            else:
                logger.warning(
                    f"Unknown WebSocket message type {message.method} received: "
                    f"{message.data}"
                )
                await _answer_failure(
                    websocket,
                    message,
                    "Message type is not processed by the server",
                    retryable=False,
                )

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception:
        logger.exception(f"Unexpected error in websocket_router for user #{user_id}")
    finally:
        # Always release subscription pumps + Redis connections, regardless of how
        # the loop exited — otherwise non-WebSocketDisconnect failures leak both.
        await manager.disconnect_socket(websocket, user_id=user_id)
        update_websocket_connections(user_id, -1)


async def _handle_request(
    manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WSMessage,
    failed_before: set[type[Exception]],
) -> bool:
    """Run the request's handler, answering a failure with an error that
    names the request and whether it may succeed if sent again. False when
    the client left meanwhile: there is no one to answer."""
    try:
        await _MSG_HANDLERS[message.method](
            connection_manager=manager,
            websocket=websocket,
            user_id=user_id,
            message=message,
        )
    except pydantic.ValidationError as e:
        logger.error(
            "Validation error while handling '%s' for user #%s: %s",
            message.method.value,
            user_id,
            loggable_validation_errors(e),
        )
        await _answer_failure(
            websocket,
            message,
            "Invalid message data. Refer to the API schema",
            retryable=False,
        )
    except ValueError as e:
        # Refused: a cap reached, or a run that isn't the user's. A full
        # shard ("Server busy") is a cap too: retried on this socket, it only
        # adds load to a full process; the next reconnect tries again.
        logger.warning(
            "Subscription rejected for user #%s on '%s': %s",
            user_id,
            message.method.value,
            e,
        )
        await _answer_failure(websocket, message, str(e), retryable=False)
    except Exception as e:
        if _is_ws_close_race(e, websocket):
            return False
        # Redis didn't confirm in time, a stream was lost, a read failed. The
        # client sends it again, so only the socket's first failure of each
        # kind is an error, which Sentry files; its retries' are warnings.
        logger.log(
            logging.WARNING if type(e) in failed_before else logging.ERROR,
            "Error while handling '%s' message for user #%s: %s",
            message.method.value,
            user_id,
            e,
        )
        failed_before.add(type(e))
        await _answer_failure(
            websocket,
            message,
            f"Could not handle '{message.method.value}'; retry",
            retryable=True,
        )
    return True


async def _answer_failure(
    websocket: WebSocket, message: WSMessage, error: str, *, retryable: bool
) -> None:
    await websocket.send_text(
        WSRequestFailed(
            error=error,
            request=WSRequestSent(method=message.method, data=message.data),
            retryable=retryable,
        ).model_dump_json()
    )


async def _refused_now(user_id: str) -> bool:
    """A lookup that fails keeps the socket; the next message asks again."""
    try:
        return await is_user_refused(user_id)
    except Exception:
        logger.warning(
            f"Closed-account lookup failed for user #{user_id}; keeping the socket"
        )
        return False


async def _banned_now(user_id: str) -> bool:
    """A lookup that fails keeps the socket; the next message asks again."""
    try:
        return await is_user_banned(user_id)
    except Exception:
        logger.warning(f"Ban lookup failed for user #{user_id}; keeping the socket")
        return False


@app.get("/")
@app.get("/health")
async def health():
    return {"status": "healthy"}


class WebsocketServer(AppProcess):
    reads_feature_flags = False

    def run(self):
        logger.info(f"CORS allow origins: {settings.config.backend_cors_allow_origins}")
        cors_params = build_cors_params(
            settings.config.backend_cors_allow_origins,
            settings.config.app_env,
        )
        server_app = CORSMiddleware(
            app=app,
            **cors_params,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        start_metrics_server(Config().websocket_server_metrics_port)
        uvicorn.run(
            server_app,
            host=Config().websocket_server_host,
            port=Config().websocket_server_port,
            ws="websockets-sansio",
            ws_max_size=WS_MAX_MESSAGE_BYTES,
            log_config=None,
            forwarded_allow_ips=uvicorn_forwarded_allow_ips(
                settings.config.trusted_proxies
            ),
        )
