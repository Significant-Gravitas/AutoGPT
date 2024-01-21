from __future__ import annotations

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class PlansTable(BaseNoSQLTable):
    table_name = "plans"
    primary_key = "plan_id"
    secondary_key = "agent_id"
    third_key = "user_id"

    from AFAAS.lib.task.plan import Plan

    async def add(self, value: dict, id: str = Plan.generate_uuid()) -> str:
        return await super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    async def update(self, plan_id: str, agent_id: str, value: dict):
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return await super().update(key=key, value=value)

    async def delete(self, plan_id: str, agent_id: str):
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return await super().delete(key=key)

    async def get(self, plan_id: str, agent_id: str) -> Plan:
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return await super().get(key=key)
