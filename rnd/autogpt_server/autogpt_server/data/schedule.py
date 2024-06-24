import json
from datetime import datetime
from typing import Optional, Any

from prisma.models import AgentExecutionSchedule

from autogpt_server.data.db import BaseDbModel


class ExecutionSchedule(BaseDbModel):
    id: str
    agent_id: str
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
    def from_db(schedule: AgentExecutionSchedule):
        return ExecutionSchedule(
            id=schedule.id,
            agent_id=schedule.agentGraphId,
            schedule=schedule.schedule,
            is_enabled=schedule.isEnabled,
            last_updated=schedule.lastUpdated.replace(tzinfo=None),
            input_data=json.loads(schedule.inputData),
        )


async def get_active_schedules(last_fetch_time: datetime) -> list[ExecutionSchedule]:
    query = AgentExecutionSchedule.prisma().find_many(
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
    await AgentExecutionSchedule.prisma().update(
        where={"id": schedule_id},
        data={"isEnabled": False}
    )


async def get_schedules(agent_id: str) -> list[ExecutionSchedule]:
    query = AgentExecutionSchedule.prisma().find_many(
        where={
            "isEnabled": True,
            "agentGraphId": agent_id,
        },
    )
    return [
        ExecutionSchedule.from_db(schedule)
        for schedule in await query
    ]


async def add_schedule(schedule: ExecutionSchedule):
    await AgentExecutionSchedule.prisma().create(
        data={
            "id": schedule.id,
            "agentGraphId": schedule.agent_id,
            "schedule": schedule.schedule,
            "isEnabled": schedule.is_enabled,
            "inputData": json.dumps(schedule.input_data),
        }
    )


async def update_schedule(schedule_id: str, is_enabled: bool):
    await AgentExecutionSchedule.prisma().update(
        where={"id": schedule_id},
        data={"isEnabled": is_enabled}
    )
