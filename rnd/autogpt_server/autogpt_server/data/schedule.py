import json
from datetime import datetime
from typing import Optional, Any

from prisma.models import AgentGraphExecutionSchedule

from autogpt_server.data.db import BaseDbModel


class ExecutionSchedule(BaseDbModel):
    graph_id: str
    schedule: str
    is_enabled: bool
    input_data: dict[str, Any]
    last_updated: Optional[datetime] = None

    def __init__(
            self,
            is_enabled: Optional[bool] = None,
            **kwargs
    ):
        if is_enabled is None:
            is_enabled = True
        super().__init__(is_enabled=is_enabled, **kwargs)

    @staticmethod
    def from_db(schedule: AgentGraphExecutionSchedule):
        return ExecutionSchedule(
            id=schedule.id,
            graph_id=schedule.agentGraphId,
            schedule=schedule.schedule,
            is_enabled=schedule.isEnabled,
            last_updated=schedule.lastUpdated.replace(tzinfo=None),
            input_data=json.loads(schedule.inputData),
        )


async def get_active_schedules(last_fetch_time: datetime) -> list[ExecutionSchedule]:
    query = AgentGraphExecutionSchedule.prisma().find_many(
        where={
            "isEnabled": True,
            "lastUpdated": {"gt": last_fetch_time}
        },
        order={"lastUpdated": "asc"}
    )
    return [
        ExecutionSchedule.from_db(schedule)
        for schedule in await query
    ]


async def disable_schedule(schedule_id: str):
    await AgentGraphExecutionSchedule.prisma().update(
        where={"id": schedule_id},
        data={"isEnabled": False}
    )


async def get_schedules(graph_id: str) -> list[ExecutionSchedule]:
    query = AgentGraphExecutionSchedule.prisma().find_many(
        where={
            "isEnabled": True,
            "agentGraphId": graph_id,
        },
    )
    return [
        ExecutionSchedule.from_db(schedule)
        for schedule in await query
    ]


async def add_schedule(schedule: ExecutionSchedule) -> ExecutionSchedule:
    obj = await AgentGraphExecutionSchedule.prisma().create(
        data={
            "id": schedule.id,
            "agentGraphId": schedule.graph_id,
            "schedule": schedule.schedule,
            "isEnabled": schedule.is_enabled,
            "inputData": json.dumps(schedule.input_data),
        }
    )
    return ExecutionSchedule.from_db(obj)


async def update_schedule(schedule_id: str, is_enabled: bool):
    await AgentGraphExecutionSchedule.prisma().update(
        where={"id": schedule_id},
        data={"isEnabled": is_enabled}
    )
