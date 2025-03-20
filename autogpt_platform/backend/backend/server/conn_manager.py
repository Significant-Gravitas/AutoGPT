from typing import Dict, Set

from fastapi import WebSocket

from backend.data.execution import (
    ExecutionEventType,
    GraphExecutionEvent,
    NodeExecutionEvent,
)
from backend.server.model import WSMessage, WSMethod

_EVENT_TYPE_TO_METHOD_MAP: dict[ExecutionEventType, WSMethod] = {
    ExecutionEventType.GRAPH_EXEC_UPDATE: WSMethod.GRAPH_EXECUTION_EVENT,
    ExecutionEventType.NODE_EXEC_UPDATE: WSMethod.NODE_EXECUTION_EVENT,
}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        for subscribers in self.subscriptions.values():
            subscribers.discard(websocket)

    async def subscribe(self, graph_id: str, graph_version: int, websocket: WebSocket):
        key = f"{graph_id}_{graph_version}"
        if key not in self.subscriptions:
            self.subscriptions[key] = set()
        self.subscriptions[key].add(websocket)

    async def unsubscribe(
        self, graph_id: str, graph_version: int, websocket: WebSocket
    ):
        key = f"{graph_id}_{graph_version}"
        if key in self.subscriptions:
            self.subscriptions[key].discard(websocket)
            if not self.subscriptions[key]:
                del self.subscriptions[key]

    async def send_execution_update(
        self, execution_event: GraphExecutionEvent | NodeExecutionEvent
    ):
        key = f"{execution_event.graph_id}_{execution_event.graph_version}"
        if key in self.subscriptions:
            message = WSMessage(
                method=_EVENT_TYPE_TO_METHOD_MAP[execution_event.event_type],
                channel=key,
                data=execution_event.model_dump(),
            ).model_dump_json()
            for connection in self.subscriptions[key]:
                await connection.send_text(message)
