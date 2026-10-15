from typing import Any

import pydantic

from backend.data.model import CredentialsMetaInput
from backend.data.schedule import ScheduleName


class ScheduleCreationRequest(pydantic.BaseModel):
    graph_version: int | None = None
    name: ScheduleName
    cron: str
    inputs: dict[str, Any]
    credentials: dict[str, CredentialsMetaInput] = pydantic.Field(default_factory=dict)
    timezone: str | None = pydantic.Field(
        default=None,
        description="User's timezone for scheduling (e.g., 'America/New_York'). If not provided, will use user's saved timezone or UTC.",
    )
    expert_id: str | None = pydantic.Field(
        default=None,
        description="Attribute this schedule (and every run it fires) to a hired expert owned by the caller. If omitted, resolved automatically when exactly one active hired expert has this graph installed as a workflow.",
    )


class ScheduleUpdateRequest(pydantic.BaseModel):
    """Fields to change on an existing schedule; omitted fields keep their value."""

    name: ScheduleName | None = None
    cron: str | None = None
    inputs: dict[str, Any] | None = None
    credentials: dict[str, CredentialsMetaInput] | None = None
    timezone: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Timezone to evaluate the cron in (e.g., 'America/New_York'). If not provided, the schedule keeps its current timezone.",
    )
