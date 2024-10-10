import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from autogpt_libs.auth import parse_jwt_token
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.data.queue import RedisEventQueue
from backend.data.user import DEFAULT_USER_ID
from backend.server.conn_manager import ConnectionManager
from backend.server.model import ExecutionSubscription, Methods, WsMessage
from backend.util.service import AppProcess
from backend.util.settings import AppEnvironment, Config, Settings

logger = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    event_queue.connect()
    manager = get_connection_manager()
    fut = asyncio.create_task(event_broadcaster(manager))
    fut.add_done_callback(lambda _: logger.info("Event broadcaster stopped"))
    yield
    event_queue.close()


docs_url = "/docs" if settings.config.app_env == AppEnvironment.LOCAL else None
app = FastAPI(lifespan=lifespan)
event_queue = RedisEventQueue()
_connection_manager = None

logger.info(f"CORS allow origins: {settings.config.backend_cors_allow_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.config.backend_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection_manager():
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


async def event_broadcaster(manager: ConnectionManager):
    while True:
        event = event_queue.get()
        if event is not None:
            await manager.send_execution_result(event)
        else:
            await asyncio.sleep(0.1)


async def authenticate_websocket(websocket: WebSocket) -> str:
    if settings.config.enable_auth.lower() == "true":
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
    else:
        return DEFAULT_USER_ID


async def handle_subscribe(
    websocket: WebSocket, manager: ConnectionManager, message: WsMessage
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
        ex_sub = ExecutionSubscription.model_validate(message.data)
        await manager.subscribe(ex_sub.graph_id, websocket)
        logger.debug(f"New execution subscription for graph {ex_sub.graph_id}")
        await websocket.send_text(
            WsMessage(
                method=Methods.SUBSCRIBE,
                success=True,
                channel=ex_sub.graph_id,
            ).model_dump_json()
        )


async def handle_unsubscribe(
    websocket: WebSocket, manager: ConnectionManager, message: WsMessage
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
        ex_sub = ExecutionSubscription.model_validate(message.data)
        await manager.unsubscribe(ex_sub.graph_id, websocket)
        logger.debug(f"Removed execution subscription for graph {ex_sub.graph_id}")
        await websocket.send_text(
            WsMessage(
                method=Methods.UNSUBSCRIBE,
                success=True,
                channel=ex_sub.graph_id,
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
            if message.method == Methods.SUBSCRIBE:
                await handle_subscribe(websocket, manager, message)

            elif message.method == Methods.UNSUBSCRIBE:
                await handle_unsubscribe(websocket, manager, message)

            elif message.method == Methods.ERROR:
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
        uvicorn.run(
            app,
            host=Config().websocket_server_host,
            port=Config().websocket_server_port,
        )
