from fastapi import WebSocket, WebSocketDisconnect
import pydantic
import typing
import enum


class Methods(enum.Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    UPDATE = "update"
    ERROR = "error"


class WsMessage(pydantic.BaseModel):
    method: Methods
    data: typing.Dict[str, typing.Any] | None = None
    success: bool | None = None
    channel: str | None = None
    error: str | None = None


class ExecutionSubscription(pydantic.BaseModel):
    channel: str
    graph_id: str
    run_id: str


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = WsMessage.model_validate_json(data)
            if message.method == Methods.SUBSCRIBE:
                print("subscribed")
                send_data = WsMessage(
                    method=Methods.SUBSCRIBE, success=True, channel="test"
                )
            elif message.method == Methods.UNSUBSCRIBE:
                print("unsubscribed")
                send_data = WsMessage(
                    method=Methods.SUBSCRIBE, success=True, channel="test"
                )
            else:
                print("Message type is not processed by the server")
                send_data = WsMessage(
                    method=Methods.ERROR,
                    success=False,
                    error="Message type is not processed by the server",
                )

            await websocket.send_text(send_data.model_dump_json())
    except WebSocketDisconnect:
        print("Client Disconnected")
