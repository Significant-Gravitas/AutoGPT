from fastapi import WebSocket, WebSocketDisconnect

from autogpt_server.server.conn_manager import ConnectionManager
from autogpt_server.server.model import ExecutionSubscription, Methods, WsMessage


async def websocket_router(websocket: WebSocket, manager: ConnectionManager):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = WsMessage.model_validate_json(data)
            if message.method == Methods.SUBSCRIBE:
                await handle_subscribe(websocket, manager, message)

            elif message.method == Methods.UNSUBSCRIBE:
                await handle_unsubscribe(websocket, manager, message)
            else:
                print("Message type is not processed by the server")
                await websocket.send_text(
                    WsMessage(
                        method=Methods.ERROR,
                        success=False,
                        error="Message type is not processed by the server",
                    ).model_dump_json()
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client Disconnected")


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
        print("subscribed")
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
        print("unsubscribed")
        await websocket.send_text(
            WsMessage(
                method=Methods.UNSUBSCRIBE,
                success=True,
                channel=ex_sub.graph_id,
            ).model_dump_json()
        )
