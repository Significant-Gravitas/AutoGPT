import enum
from typing import Any, Literal, Optional

import pydantic
from prisma.enums import OnboardingStep

from backend.data.auth.api_key import APIKeyInfo, APIKeyPermission
from backend.data.graph import Graph
from backend.util.timezone_name import TimeZoneName


class WSMethod(enum.Enum):
    SUBSCRIBE_GRAPH_EXEC = "subscribe_graph_execution"
    SUBSCRIBE_GRAPH_EXECS = "subscribe_graph_executions"
    UNSUBSCRIBE = "unsubscribe"
    GRAPH_EXECUTION_EVENT = "graph_execution_event"
    NODE_EXECUTION_EVENT = "node_execution_event"
    NOTIFICATION = "notification"
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


class WSSubscribeGraphExecutionsRequest(pydantic.BaseModel):
    graph_id: str


GraphCreationSource = Literal["builder", "upload"]
GraphExecutionSource = Literal["builder", "library", "onboarding"]


class CreateGraph(pydantic.BaseModel):
    graph: Graph
    source: GraphCreationSource | None = None


class CreateAPIKeyRequest(pydantic.BaseModel):
    name: str
    permissions: list[APIKeyPermission]
    description: Optional[str] = None


class CreateAPIKeyResponse(pydantic.BaseModel):
    api_key: APIKeyInfo
    plain_text_key: str


class SetGraphActiveVersion(pydantic.BaseModel):
    active_graph_version: int


class UpdatePermissionsRequest(pydantic.BaseModel):
    permissions: list[APIKeyPermission]


class RequestTopUp(pydantic.BaseModel):
    credit_amount: int


class UploadFileResponse(pydantic.BaseModel):
    file_uri: str
    file_name: str
    size: int
    content_type: str
    expires_in_hours: int


class TimezoneResponse(pydantic.BaseModel):
    # Allow "not-set" as a special value, or any valid IANA timezone
    timezone: TimeZoneName | str


class UpdateTimezoneRequest(pydantic.BaseModel):
    timezone: TimeZoneName


class NotificationPayload(pydantic.BaseModel):
    type: str
    event: str

    model_config = pydantic.ConfigDict(extra="allow")


class OnboardingNotificationPayload(NotificationPayload):
    step: OnboardingStep | None
