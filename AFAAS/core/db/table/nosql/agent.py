from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class AgentsTable(BaseNoSQLTable):
    table_name = "agents"
    primary_key = "agent_id"
    secondary_key = "user_id"
    third_key = "agent_type"

    if TYPE_CHECKING:
        from AFAAS.interfaces.agent.main import BaseAgent

    async def add(self, value: dict, id: str = "A" + str(uuid.uuid4())) -> str:
        return await super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    async def update(self, agent_id: str, user_id: str, value: dict):
        key = AgentsTable.Key(
            primary_key=str(agent_id),
            secondary_key=str(user_id),
        )
        return await super().update(key=key, value=value)

    async def delete(self, agent_id: str, user_id: str):
        key = AgentsTable.Key(
            primary_key=str(agent_id),
            secondary_key=str(user_id),
        )
        return await super().delete(key=key)

    async def get(self, agent_id: str, user_id: str) -> BaseAgent:
        key = AgentsTable.Key(
            primary_key=str(agent_id),
            secondary_key=str(user_id),
        )
        return await super().get(key=key)
