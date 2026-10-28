import enum
from datetime import datetime
from typing import Any, Literal, Optional

import pydantic

from backend.data.graph import Graph
from backend.data.onboarding_steps import OnboardingStep
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


class SetGraphActiveVersion(pydantic.BaseModel):
    active_graph_version: int


class RequestTopUp(pydantic.BaseModel):
    credit_amount: int
    surface: Optional[Literal["billing"]] = pydantic.Field(
        default=None, description="Where the top-up was started; analytics only."
    )


class CloudStorageUploadResponse(pydantic.BaseModel):
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


# Every terms and privacy policy version the signup page can show. Add a new
# one here, and deploy it, before the frontend's TERMS_VERSION (lib/legal.ts)
# moves to it; routes_test fails if the frontend's version is missing here.
RECOGNIZED_TERMS_VERSIONS = frozenset({"2026-10"})


class RecordUserConsentRequest(pydantic.BaseModel):
    # The frontend's TERMS_VERSION, e.g. "2026-10", or "2026-10-15" for a
    # second change in one month. ASCII digits only: `\d` would also take
    # other scripts' digits.
    terms_version: str = pydantic.Field(
        min_length=1, max_length=32, pattern=r"^[0-9]{4}-[0-9]{2}(-[0-9]{2})?$"
    )
    marketing_opt_out: bool

    @pydantic.field_validator("terms_version")
    @classmethod
    def _recognized(cls, terms_version: str) -> str:
        """Only a version the signup page has shown can be recorded, so a
        caller cannot claim acceptance of terms that were never offered."""
        if terms_version not in RECOGNIZED_TERMS_VERSIONS:
            raise ValueError("Unrecognized terms version")
        return terms_version


class UserConsentResponse(pydantic.BaseModel):
    terms_accepted_at: Optional[datetime] = None
    terms_version: Optional[str] = None
    marketing_opt_out_at: Optional[datetime] = None
    marketing_opt_out_source: Optional[str] = None


class NotificationPayload(pydantic.BaseModel):
    type: str
    event: str

    model_config = pydantic.ConfigDict(extra="allow")


class OnboardingNotificationPayload(NotificationPayload):
    # Typed enum: notifications only fire on fresh completions, where ``step`` is
    # always a current ``OnboardingStep`` (or ``None`` for ``increment_runs``).
    # Legacy step names live only in stored rows, never in emitted notifications.
    step: OnboardingStep | None


class CopilotCompletionPayload(NotificationPayload):
    session_id: str
    status: Literal["completed", "failed"]
