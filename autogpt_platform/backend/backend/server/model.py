import enum
import typing

import pydantic

import backend.data.graph
from backend.data.api_key import APIKeyPermission, APIKeyWithoutHash


class Methods(enum.Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    EXECUTION_EVENT = "execution_event"
    ERROR = "error"


class WsMessage(pydantic.BaseModel):
    method: Methods
    data: typing.Dict[str, typing.Any] | list[typing.Any] | None = None
    success: bool | None = None
    channel: str | None = None
    error: str | None = None


class ExecutionSubscription(pydantic.BaseModel):
    graph_id: str


class SubscriptionDetails(pydantic.BaseModel):
    event_type: str
    channel: str
    graph_id: str


class CreateGraph(pydantic.BaseModel):
    template_id: str | None = None
    template_version: int | None = None
    graph: backend.data.graph.Graph | None = None


class CreateAPIKeyRequest(pydantic.BaseModel):
    name: str
    permissions: typing.List[APIKeyPermission]
    description: typing.Optional[str] = None


class CreateAPIKeyResponse(pydantic.BaseModel):
    api_key: APIKeyWithoutHash
    plain_text_key: str


class SetGraphActiveVersion(pydantic.BaseModel):
    active_graph_version: int


class UpdatePermissionsRequest(pydantic.BaseModel):
    permissions: typing.List[APIKeyPermission]
