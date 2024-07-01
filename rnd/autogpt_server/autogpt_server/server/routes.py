from fastapi import WebSocket, WebSocketDisconnect, HTTPException
import pydantic
import typing
import enum
from autogpt_server.data import execution
from autogpt_server.data.graph import (
    get_graph,
)
from datetime import datetime


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


class SubscriptionDetails(pydantic.BaseModel):
    event_type: str
    channel: str
    graph_id: str
    run_id: str
    last_start_time: datetime | None = None


async def get_executions(graph_id: str, run_id: str) -> list[execution.ExecutionResult]:
    graph = await get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Agent #{graph_id} not found.")

    return await execution.get_executions(run_id)


async def subscription_poller(
    subscriptions: list[SubscriptionDetails], websocket: WebSocket
) -> None:
    for sub in subscriptions:
        sub: SubscriptionDetails = sub
        try:
            ex = await get_executions(sub.graph_id, sub.run_id)
        except Exception as ex:
            await websocket.send_text(
                WsMessage(
                    method=Methods.ERROR,
                    success=False,
                    channel=sub.channel,
                    error="Unable to find execution",
                ).model_dump_json()
            )
            return
        for event in ex:
            if not event.start_time:
                await websocket.send_text(
                    WsMessage(
                        method=Methods.ERROR,
                        success=False,
                        channel=sub.channel,
                        error="Start times not set",
                    ).model_dump_json()
                )
            # Only send events we have not already sent before
            # FIX: We are using time as a proxy as we dont have a sequential number to check
            if not sub.last_start_time or event.start_time > sub.last_start_time:  # type: ignore
                sub.last_start_time = event.start_time
                await websocket.send_text(
                    WsMessage(
                        method=Methods.UPDATE,
                        channel=sub.channel,
                        data=event.model_dump(),
                    ).model_dump_json()
                )


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    subscriptions = []
    try:
        while True:
            await subscription_poller(subscriptions, websocket)
            data = await websocket.receive_text()
            message = WsMessage.model_validate_json(data)
            if message.method == Methods.SUBSCRIBE:
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
                    subscriptions.append(
                        SubscriptionDetails(
                            event_type="ExecutionUpdates",
                            channel=ex_sub.channel,
                            run_id=ex_sub.run_id,
                            graph_id=ex_sub.graph_id,
                        )
                    )
                    print("subscribed")
                    await websocket.send_text(
                        WsMessage(
                            method=Methods.SUBSCRIBE,
                            success=True,
                            channel=ex_sub.channel,
                        ).model_dump_json()
                    )

            elif message.method == Methods.UNSUBSCRIBE:
                print("unsubscribed")
                await websocket.send_text(
                    WsMessage(
                        method=Methods.UNSUBSCRIBE, success=True, channel="test"
                    ).model_dump_json()
                )
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
        print("Client Disconnected")
