from typing import Any, Optional

import pydantic

from backend.data.model import CredentialsMetaInput


class ScheduleCreationRequest(pydantic.BaseModel):
    graph_version: Optional[int] = None
    name: str
    cron: str
    inputs: dict[str, Any]
    credentials: dict[str, CredentialsMetaInput] = pydantic.Field(default_factory=dict)
    timezone: Optional[str] = pydantic.Field(
        default=None,
        description="User's timezone for scheduling (e.g., 'America/New_York'). If not provided, will use user's saved timezone or UTC.",
    )
    expert_id: Optional[str] = pydantic.Field(
        default=None,
        description="Attribute this schedule (and every run it fires) to a hired expert owned by the caller. If omitted, resolved automatically when exactly one active hired expert has this graph installed as a workflow.",
    )
