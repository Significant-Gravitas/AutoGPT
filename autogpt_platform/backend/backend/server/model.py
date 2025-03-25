import enum
from typing import Any, Optional

import pydantic

from backend.data.api_key import APIKeyPermission, APIKeyWithoutHash
from backend.data.graph import Graph


class WSMethod(enum.Enum):
    SUBSCRIBE_GRAPH_EXEC = "subscribe_graph_execution"
    UNSUBSCRIBE = "unsubscribe"
    GRAPH_EXECUTION_EVENT = "graph_execution_event"
    NODE_EXECUTION_EVENT = "node_execution_event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class WSMessage(pydantic.BaseModel):
    method: WSMethod
    data: Optional[dict[str, Any] | list[Any] | str] = None
    success: bool | None = None
    channel: str | None = None
    error: str | None = None


class WSSubscribeGraphExecutionRequest(pydantic.BaseModel):
    graph_exec_id: str


class ExecuteGraphResponse(pydantic.BaseModel):
    graph_exec_id: str


class CreateGraph(pydantic.BaseModel):
    graph: Graph


class CreateAPIKeyRequest(pydantic.BaseModel):
    name: str
    permissions: list[APIKeyPermission]
    description: Optional[str] = None


class CreateAPIKeyResponse(pydantic.BaseModel):
    api_key: APIKeyWithoutHash
    plain_text_key: str


class SetGraphActiveVersion(pydantic.BaseModel):
    active_graph_version: int


class UpdatePermissionsRequest(pydantic.BaseModel):
    permissions: list[APIKeyPermission]


class Pagination(pydantic.BaseModel):
    total_items: int = pydantic.Field(
        description="Total number of items.", examples=[42]
    )
    total_pages: int = pydantic.Field(
        description="Total number of pages.", examples=[2]
    )
    current_page: int = pydantic.Field(
        description="Current_page page number.", examples=[1]
    )
    page_size: int = pydantic.Field(
        description="Number of items per page.", examples=[25]
    )


class RequestTopUp(pydantic.BaseModel):
    credit_amount: int
