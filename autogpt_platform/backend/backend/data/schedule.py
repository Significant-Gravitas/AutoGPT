from datetime import datetime
from typing import Optional

from prisma.models import AgentGraphExecutionSchedule

from backend.data.block import BlockInput
from backend.data.db import BaseDbModel
from backend.util import json


class ExecutionSchedule(BaseDbModel):
    graph_id: str
    user_id: str
    graph_version: int
    schedule: str
    is_enabled: bool
    input_data: BlockInput
    last_updated: Optional[datetime] = None

    def __init__(self, is_enabled: Optional[bool] = None, **kwargs):
        kwargs["is_enabled"] = (is_enabled is None) or is_enabled
        super().__init__(**kwargs)

    @staticmethod
    def from_db(schedule: AgentGraphExecutionSchedule):
        return ExecutionSchedule(
            id=schedule.id,
            graph_id=schedule.agentGraphId,
            user_id=schedule.userId,
            graph_version=schedule.agentGraphVersion,
            schedule=schedule.schedule,
            is_enabled=schedule.isEnabled,
            last_updated=schedule.lastUpdated.replace(tzinfo=None),
            input_data=json.loads(schedule.inputData),
        )


async def get_active_schedules(last_fetch_time: datetime) -> list[ExecutionSchedule]:
    query = AgentGraphExecutionSchedule.prisma().find_many(
        where={"isEnabled": True, "lastUpdated": {"gt": last_fetch_time}},
        order={"lastUpdated": "asc"},
    )
    return [ExecutionSchedule.from_db(schedule) for schedule in await query]


async def disable_schedule(schedule_id: str):
    await AgentGraphExecutionSchedule.prisma().update(
        where={"id": schedule_id}, data={"isEnabled": False}
    )


async def get_schedules(graph_id: str, user_id: str) -> list[ExecutionSchedule]:
    query = AgentGraphExecutionSchedule.prisma().find_many(
        where={
            "isEnabled": True,
            "agentGraphId": graph_id,
            "userId": user_id,
        },
    )
    return [ExecutionSchedule.from_db(schedule) for schedule in await query]


async def add_schedule(schedule: ExecutionSchedule) -> ExecutionSchedule:
    obj = await AgentGraphExecutionSchedule.prisma().create(
        data={
            "id": schedule.id,
            "userId": schedule.user_id,
            "agentGraphId": schedule.graph_id,
            "agentGraphVersion": schedule.graph_version,
            "schedule": schedule.schedule,
            "isEnabled": schedule.is_enabled,
            "inputData": json.dumps(schedule.input_data),
        }
    )
    return ExecutionSchedule.from_db(obj)


async def update_schedule(schedule_id: str, is_enabled: bool, user_id: str):
    await AgentGraphExecutionSchedule.prisma().update(
        where={"id": schedule_id}, data={"isEnabled": is_enabled}
    )
