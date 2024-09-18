import asyncio
import logging

import uvicorn
from autogpt_libs.auth import parse_jwt_token
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from autogpt_server.data.queue import AsyncRedisEventQueue
from autogpt_server.data.user import DEFAULT_USER_ID
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import ExecutionSubscription, Methods, WsMessage
from autogpt_server.util.service import AppProcess
from autogpt_server.util.settings import Config, Settings

logger = logging.getLogger(__name__)
settings = Settings()

app = FastAPI()
event_queue = AsyncRedisEventQueue()
_connection_manager = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://dev-builder.agpt.co",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection_manager():
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


@app.on_event("startup")
async def startup_event():
    await event_queue.connect()
    manager = get_connection_manager()
    asyncio.create_task(event_broadcaster(manager))


@app.on_event("shutdown")
async def shutdown_event():
    await event_queue.close()


async def event_broadcaster(manager: ConnectionManager):
    while True:
        event = await event_queue.get()
        if event is not None:
            await manager.send_execution_result(event)


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
        uvicorn.run(app, host="0.0.0.0", port=Config().websocket_server_port)
