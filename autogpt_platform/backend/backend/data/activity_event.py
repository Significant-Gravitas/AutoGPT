"""Append-only log of work agents performed on the user's behalf.

Written best-effort from the copilot tool layer and the graph executor,
read by the Home "Recent work" feed. Emitters must swallow their own
failures — recording the work can never be allowed to break the work.
"""

import enum
from datetime import datetime
from typing import Any, Literal, cast

import prisma.models
from pydantic import BaseModel, Field

from backend.util.json import SafeJson

ActivityEventCategory = Literal["FILE", "INTEGRATION", "RUN", "SCHEDULE"]


class ActivityEventDraft(BaseModel):
    """What an emitter knows at the moment the work happens.

    Ownership (user id) and creation time are stamped by the write, not the
    emitter. Attribution fields are optional so one shape serves copilot
    tools (session + expert) and the executor (graph + node) alike.
    """

    category: ActivityEventCategory
    event_type: str
    title: str
    organization_id: str | None = None
    expert_id: str | None = None
    session_id: str | None = None
    graph_exec_id: str | None = None
    node_exec_id: str | None = None
    schedule_id: str | None = None
    provider: str | None = None
    object_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ActivityEvent(ActivityEventDraft):
    id: str
    user_id: str
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.ActivityEvent) -> "ActivityEvent":
        # Depending on client generation settings the enum column comes back
        # as a real enum locally but a plain string in the service process.
        raw_category = row.category
        if isinstance(raw_category, enum.Enum):
            raw_category = raw_category.value
        return cls(
            id=row.id,
            user_id=row.userId,
            created_at=row.createdAt,
            organization_id=row.organizationId,
            expert_id=row.expertId,
            session_id=row.sessionId,
            graph_exec_id=row.graphExecId,
            node_exec_id=row.nodeExecId,
            schedule_id=row.scheduleId,
            category=cast(ActivityEventCategory, raw_category),
            event_type=row.eventType,
            provider=row.provider,
            object_id=row.objectId,
            title=row.title,
            data=dict(row.data) if row.data else {},
        )


async def create_activity_event(
    user_id: str, draft: ActivityEventDraft
) -> ActivityEvent:
    row = await prisma.models.ActivityEvent.prisma().create(
        data={
            "userId": user_id,
            "organizationId": draft.organization_id,
            "expertId": draft.expert_id,
            "sessionId": draft.session_id,
            "graphExecId": draft.graph_exec_id,
            "nodeExecId": draft.node_exec_id,
            "scheduleId": draft.schedule_id,
            "category": draft.category,
            "eventType": draft.event_type,
            "provider": draft.provider,
            "objectId": draft.object_id,
            "title": draft.title,
            "data": SafeJson(draft.data),
        }
    )
    return ActivityEvent.from_db(row)


async def list_activity_events(
    user_id: str,
    since: datetime,
    categories: list[ActivityEventCategory] | None = None,
    limit: int = 200,
) -> list[ActivityEvent]:
    where: dict[str, Any] = {"userId": user_id, "createdAt": {"gte": since}}
    if categories:
        where["category"] = {"in": categories}
    rows = await prisma.models.ActivityEvent.prisma().find_many(
        where=where, order={"createdAt": "desc"}, take=limit
    )
    return [ActivityEvent.from_db(row) for row in rows]
