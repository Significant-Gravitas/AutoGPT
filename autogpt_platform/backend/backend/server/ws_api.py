import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from autogpt_libs.auth import parse_jwt_token
from autogpt_libs.utils.cache import thread_cached
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from backend.data import redis
from backend.data.execution import AsyncRedisExecutionEventBus
from backend.data.user import DEFAULT_USER_ID
from backend.server.conn_manager import ConnectionManager
from backend.server.model import ExecutionSubscription, Methods, WsMessage
from backend.util.service import AppProcess, get_service_client
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


def get_connection_manager():
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


@thread_cached
def get_db_client():
    from backend.executor import DatabaseManager

    return get_service_client(DatabaseManager)


async def event_broadcaster(manager: ConnectionManager):
    try:
        redis.connect()
        event_queue = AsyncRedisExecutionEventBus()
        async for event in event_queue.listen():
            await manager.send_execution_result(event)
    except Exception as e:
        logger.exception(f"Event broadcaster error: {e}")
        raise
    finally:
        redis.disconnect()


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


async def handle_subscribe(
    connection_manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WsMessage,
):
    if not message.data:
        await websocket.send_text(
            WsMessage(
                method=Methods.ERROR,
                success=False,
                error="Subscription data missing",
            ).model_dump_json()
        )
    else:
        sub_req = ExecutionSubscription.model_validate(message.data)

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

        await connection_manager.subscribe(
            user_id=user_id,
            graph_id=sub_req.graph_id,
            graph_version=sub_req.graph_version,
            websocket=websocket,
        )
        logger.debug(
            f"New execution subscription for user #{user_id} "
            f"graph #{sub_req.graph_id}v{sub_req.graph_version}"
        )
        await websocket.send_text(
            WsMessage(
                method=Methods.SUBSCRIBE,
                success=True,
                channel=f"{user_id}_{sub_req.graph_id}_{sub_req.graph_version}",
            ).model_dump_json()
        )


async def handle_unsubscribe(
    connection_manager: ConnectionManager,
    websocket: WebSocket,
    user_id: str,
    message: WsMessage,
):
    if not message.data:
        await websocket.send_text(
            WsMessage(
                method=Methods.ERROR,
                success=False,
                error="Subscription data missing",
            ).model_dump_json()
        )
    else:
        unsub_req = ExecutionSubscription.model_validate(message.data)
        await connection_manager.unsubscribe(
            user_id=user_id,
            graph_id=unsub_req.graph_id,
            graph_version=unsub_req.graph_version,
            websocket=websocket,
        )
        logger.debug(
            f"Removed execution subscription for user #{user_id} "
            f"graph #{unsub_req.graph_id}v{unsub_req.graph_version}"
        )
        await websocket.send_text(
            WsMessage(
                method=Methods.UNSUBSCRIBE,
                success=True,
                channel=f"{unsub_req.graph_id}_{unsub_req.graph_version}",
            ).model_dump_json()
        )


@app.get("/")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_router(
    websocket: WebSocket, manager: ConnectionManager = Depends(get_connection_manager)
):
    user_id = await authenticate_websocket(websocket)
    if not user_id:
        return
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = WsMessage.model_validate_json(data)

            if message.method == Methods.HEARTBEAT:
                await websocket.send_json(
                    {"method": Methods.HEARTBEAT.value, "data": "pong", "success": True}
                )
                continue

            try:
                if message.method == Methods.SUBSCRIBE:
                    await handle_subscribe(
                        connection_manager=manager,
                        websocket=websocket,
                        user_id=user_id,
                        message=message,
                    )
                    continue

                elif message.method == Methods.UNSUBSCRIBE:
                    await handle_unsubscribe(
                        connection_manager=manager,
                        websocket=websocket,
                        user_id=user_id,
                        message=message,
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Error while handling '{message.method}' message "
                    f"for user #{user_id}: {e}"
                )
                continue

            if message.method == Methods.ERROR:
                logger.error(f"WebSocket Error message received: {message.data}")

            else:
                logger.warning(
                    f"Unknown WebSocket message type {message.method} received: "
                    f"{message.data}"
                )
                await websocket.send_text(
                    WsMessage(
                        method=Methods.ERROR,
                        success=False,
                        error="Message type is not processed by the server",
                    ).model_dump_json()
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.debug("WebSocket client disconnected")


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
        )
