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

    async def connect_socket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect_socket(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        for subscribers in self.subscriptions.values():
            subscribers.discard(websocket)

    async def subscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str:
        key = _graph_exec_channel_key(user_id, graph_exec_id)
        if key not in self.subscriptions:
            self.subscriptions[key] = set()
        self.subscriptions[key].add(websocket)
        return key

    async def unsubscribe(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        key = _graph_exec_channel_key(user_id, graph_exec_id)
        if key in self.subscriptions:
            self.subscriptions[key].discard(websocket)
            if not self.subscriptions[key]:
                del self.subscriptions[key]
            return key
        return None

    async def send_execution_update(
        self, exec_event: GraphExecutionEvent | NodeExecutionEvent
    ) -> int:
        graph_exec_id = (
            exec_event.id
            if isinstance(exec_event, GraphExecutionEvent)
            else exec_event.graph_exec_id
        )
        key = _graph_exec_channel_key(exec_event.user_id, graph_exec_id)

        n_sent = 0
        if key in self.subscriptions:
            message = WSMessage(
                method=_EVENT_TYPE_TO_METHOD_MAP[exec_event.event_type],
                channel=key,
                data=exec_event.model_dump(),
            ).model_dump_json()
            for connection in self.subscriptions[key]:
                await connection.send_text(message)
                n_sent += 1

        return n_sent


def _graph_exec_channel_key(user_id: str, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"
