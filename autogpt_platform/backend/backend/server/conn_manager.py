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
        return await self._subscribe(
            _graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id), websocket
        )

    async def subscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str:
        return await self._subscribe(
            _graph_execs_channel_key(user_id, graph_id=graph_id), websocket
        )

    async def unsubscribe_graph_exec(
        self, *, user_id: str, graph_exec_id: str, websocket: WebSocket
    ) -> str | None:
        return await self._unsubscribe(
            _graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id), websocket
        )

    async def unsubscribe_graph_execs(
        self, *, user_id: str, graph_id: str, websocket: WebSocket
    ) -> str | None:
        return await self._unsubscribe(
            _graph_execs_channel_key(user_id, graph_id=graph_id), websocket
        )

    async def send_execution_update(
        self, exec_event: GraphExecutionEvent | NodeExecutionEvent
    ) -> int:
        graph_exec_id = (
            exec_event.id
            if isinstance(exec_event, GraphExecutionEvent)
            else exec_event.graph_exec_id
        )

        n_sent = 0

        channels: set[str] = {
            # Send update to listeners for this graph execution
            _graph_exec_channel_key(exec_event.user_id, graph_exec_id=graph_exec_id)
        }
        if isinstance(exec_event, GraphExecutionEvent):
            # Send update to listeners for all executions of this graph
            channels.add(
                _graph_execs_channel_key(
                    exec_event.user_id, graph_id=exec_event.graph_id
                )
            )

        for channel in channels.intersection(self.subscriptions.keys()):
            message = WSMessage(
                method=_EVENT_TYPE_TO_METHOD_MAP[exec_event.event_type],
                channel=channel,
                data=exec_event.model_dump(),
            ).model_dump_json()
            for connection in self.subscriptions[channel]:
                await connection.send_text(message)
                n_sent += 1

        return n_sent

    async def _subscribe(self, channel_key: str, websocket: WebSocket) -> str:
        if channel_key not in self.subscriptions:
            self.subscriptions[channel_key] = set()
        self.subscriptions[channel_key].add(websocket)
        return channel_key

    async def _unsubscribe(self, channel_key: str, websocket: WebSocket) -> str | None:
        if channel_key in self.subscriptions:
            self.subscriptions[channel_key].discard(websocket)
            if not self.subscriptions[channel_key]:
                del self.subscriptions[channel_key]
            return channel_key
        return None


def _graph_exec_channel_key(user_id: str, *, graph_exec_id: str) -> str:
    return f"{user_id}|graph_exec#{graph_exec_id}"


def _graph_execs_channel_key(user_id: str, *, graph_id: str) -> str:
    return f"{user_id}|graph#{graph_id}|executions"
