import asyncio

from autogpt_libs.auth.jwt_utils import parse_jwt_token
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import autogpt_server.server.ws_api as ws_api
from autogpt_server.data import user as user_db
from autogpt_server.data.execution import ExecutionResult
from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import Methods, WsMessage
from autogpt_server.util.settings import Settings

settings = Settings()


class WebSocketServer:
    def __init__(self):
        self.app = FastAPI()
        self.manager = ConnectionManager()
        self.event_queue = asyncio.Queue()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_router(websocket)

    async def authenticate_websocket(self, websocket: WebSocket) -> str:
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
            return user_db.DEFAULT_USER_ID

    async def websocket_router(self, websocket: WebSocket):
        user_id = await self.authenticate_websocket(websocket)
        if not user_id:
            return
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                message = WsMessage.model_validate_json(data)
                if message.method == Methods.SUBSCRIBE:
                    await ws_api.handle_subscribe(websocket, self.manager, message)
                elif message.method == Methods.UNSUBSCRIBE:
                    await ws_api.handle_unsubscribe(websocket, self.manager, message)
                elif message.method == Methods.ERROR:
                    print("WebSocket Error message received:", message.data)
                else:
                    print(
                        f"Message type {message.method} is not processed by the server"
                    )
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.ERROR,
                            success=False,
                            error="Message type is not processed by the server",
                        ).model_dump_json()
                    )
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            print("Client Disconnected")

    async def event_broadcaster(self):
        while True:
            event = await self.event_queue.get()
            await self.manager.send_execution_result(event)

    async def add_execution_update(self, execution_result: ExecutionResult):
        await self.event_queue.put(execution_result)

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        import uvicorn

        asyncio.create_task(self.event_broadcaster())
        uvicorn.run(self.app, host=host, port=port)
