from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .base import BaseNoSQLTable


class PlansTable(BaseNoSQLTable):
    table_name = "plans"
    primary_key = "plan_id"
    secondary_key = "agent_id"
    third_key = "user_id"

    from AFAAS.app.lib.task.plan import Plan

    def add(self, value: dict, id: str = Plan.generate_uuid()) -> str:
        return super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    def update(self, plan_id: str, agent_id: str, value: dict):
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().update(key=key, value=value)

    def delete(self, plan_id: str, agent_id: str):
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().delete(key=key)

    def get(self, plan_id: str, agent_id: str) -> Plan:
        key = PlansTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().get(key=key)
