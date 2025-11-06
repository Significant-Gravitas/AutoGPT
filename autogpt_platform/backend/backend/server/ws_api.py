import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Protocol

import pydantic
import uvicorn
from autogpt_libs.auth.jwt_utils import parse_jwt_token
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from backend.data.execution import AsyncRedisExecutionEventBus
from backend.data.user import DEFAULT_USER_ID
from backend.monitoring.instrumentation import (
    instrument_fastapi,
    update_websocket_connections,
)
from backend.server.conn_manager import ConnectionManager
from backend.server.model import (
    WSMessage,
    WSMethod,
    WSSubscribeGraphExecutionRequest,
    WSSubscribeGraphExecutionsRequest,
)
from backend.util.retry import continuous_retry
from backend.util.service import AppProcess
from backend.util.settings import AppEnvironment, Config, Settings

logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = get_connection_manager()
    fut = asyncio.create_task(event_broadcaster(manager))
    fut.add_done_callback(lambda _: logger.info("Event broadcaster stopped"))
    yield


docs_url = "/docs" if settings.config.app_env == AppEnvironment.LOCAL else None
app = FastAPI(lifespan=lifespan, docs_url=docs_url)
_connection_manager = None

# Add Prometheus instrumentation
instrument_fastapi(
    app,
    service_name="websocket-server",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=settings.config.app_env == AppEnvironment.LOCAL,
)


def get_connection_manager():
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


@continuous_retry()
async def event_broadcaster(manager: ConnectionManager):
    event_queue = AsyncRedisExecutionEventBus()
    async for event in event_queue.listen("*"):
        await manager.send_execution_update(event)


async def authenticate_websocket(websocket: WebSocket) -> str:
    if not settings.config.enable_auth:
        return DEFAULT_USER_ID

    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return ""

    try:
        payload = parse_jwt_token(token)
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=4002, reason="Invalid token")
            return ""
        return user_id
    except ValueError:
        await websocket.close(code=4003, reason="Invalid token")
        return ""


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
    await manager.connect_socket(websocket)

    # Track WebSocket connection
    update_websocket_connections(user_id, 1)

    try:
        while True:
            data = await websocket.receive_text()
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
            except Exception as e:
                logger.error(
                    f"Error while handling '{message.method.value}' message "
                    f"for user #{user_id}: {e}"
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
        manager.disconnect_socket(websocket)
        logger.debug("WebSocket client disconnected")
    finally:
        update_websocket_connections(user_id, -1)


@app.get("/")
async def health():
    return {"status": "healthy"}


class WebsocketServer(AppProcess):
    def run(self):
        logger.info(f"CORS allow origins: {settings.config.backend_cors_allow_origins}")
        server_app = CORSMiddleware(
            app=app,
            allow_origins=settings.config.backend_cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        uvicorn.run(
            server_app,
            host=Config().websocket_server_host,
            port=Config().websocket_server_port,
            ws="websockets-sansio",
            log_config=None,
        )
