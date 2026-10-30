import logging
from contextlib import asynccontextmanager
from typing import Protocol

import pydantic
import uvicorn
from autogpt_libs.auth.jwt_utils import parse_jwt_token_async
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from backend.api.conn_manager import ConnectionManager, _is_ws_close_race
from backend.api.model import (
    WSMessage,
    WSMethod,
    WSSubscribeGraphExecutionRequest,
    WSSubscribeGraphExecutionsRequest,
)
from backend.api.utils.cors import build_cors_params
from backend.data import redis_client
from backend.data.user import is_user_banned
from backend.monitoring.instrumentation import (
    instrument_fastapi,
    start_metrics_server,
    update_websocket_connections,
)
from backend.util.service import AppProcess
from backend.util.settings import AppEnvironment, Config, Settings

logger = logging.getLogger(__name__)
settings = Settings()

ACCOUNT_BANNED_CLOSE_CODE = 4004


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No Prisma: reads go through the DatabaseManager (db_accessors), so this
    # server needs no database URL. Eager Redis connect to fail fast.
    await redis_client.get_redis_async()
    try:
        yield
    finally:
        # Silences asyncio's "Unclosed ClusterNode" GC warning at shutdown.
        try:
            await redis_client.disconnect_async()
        except Exception:
            logger.warning("redis_client.disconnect_async failed", exc_info=True)


docs_url = "/docs" if settings.config.app_env == AppEnvironment.LOCAL else None
app = FastAPI(lifespan=lifespan, docs_url=docs_url)
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
        await websocket.close(code=4001, reason="Missing authentication token")
        return ""

    try:
        payload = await parse_jwt_token_async(token)
    except ValueError:
        await websocket.close(code=4003, reason="Invalid token")
        return ""
    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=4002, reason="Invalid token")
        return ""

    # Refused when the ban can't be read, as a request to a signed-in route is;
    # 1011 rather than the ban's code, so an outage never reads as a ban.
    try:
        banned = await is_user_banned(user_id)
    except Exception:
        logger.warning(f"Ban lookup failed for user #{user_id}; refusing the socket")
        await websocket.close(code=1011, reason="Try again")
        return ""
    if banned:
        await websocket.close(code=ACCOUNT_BANNED_CLOSE_CODE, reason="Banned")
        return ""
    return user_id


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
        await websocket.send_text(
            WSMessage(
                method=WSMethod.ERROR,
                success=False,
                error="Subscription data missing",
            ).model_dump_json()
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
        await websocket.send_text(
            WSMessage(
                method=WSMethod.ERROR,
                success=False,
                error="Subscription data missing",
            ).model_dump_json()
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
    await manager.connect_socket(websocket, user_id=user_id)

    # Track WebSocket connection
    update_websocket_connections(user_id, 1)

    try:
        while True:
            data = await websocket.receive_text()
            # The socket outlives its token, so a ban ends it at the next message.
            if await _banned_now(user_id):
                await websocket.close(code=ACCOUNT_BANNED_CLOSE_CODE, reason="Banned")
                break
            try:
                message = WSMessage.model_validate_json(data)
            except pydantic.ValidationError as e:
                logger.error(
                    "Invalid WebSocket message from user #%s: %s",
                    user_id,
                    e,
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error=("Invalid message format. Review the schema and retry"),
                    ).model_dump_json()
                )
                continue

            try:
                if message.method in _MSG_HANDLERS:
                    await _MSG_HANDLERS[message.method](
                        connection_manager=manager,
                        websocket=websocket,
                        user_id=user_id,
                        message=message,
                    )
                    continue
            except pydantic.ValidationError as e:
                logger.error(
                    "Validation error while handling '%s' for user #%s: %s",
                    message.method.value,
                    user_id,
                    e,
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error="Invalid message data. Refer to the API schema",
                    ).model_dump_json()
                )
                continue
            except ValueError as e:
                logger.warning(
                    "Subscription rejected for user #%s on '%s': %s",
                    user_id,
                    message.method.value,
                    e,
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error=str(e),
                    ).model_dump_json()
                )
                continue
            except Exception as e:
                # The client left mid-handler: there is no one to reply to.
                if _is_ws_close_race(e, websocket):
                    break
                logger.error(
                    f"Error while handling '{message.method.value}' message "
                    f"for user #{user_id}: {e}"
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error=f"Could not handle '{message.method.value}'; retry",
                    ).model_dump_json()
                )
                continue

            if message.method == WSMethod.ERROR:
                logger.error(f"WebSocket Error message received: {message.data}")

            else:
                logger.warning(
                    f"Unknown WebSocket message type {message.method} received: "
                    f"{message.data}"
                )
                await websocket.send_text(
                    WSMessage(
                        method=WSMethod.ERROR,
                        success=False,
                        error="Message type is not processed by the server",
                    ).model_dump_json()
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
            log_config=None,
        )
