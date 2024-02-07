from __future__ import annotations

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class MessagesUserAgentTable(BaseNoSQLTable):
    table_name = "message_agent_user"  #  FIXME:0.0.4 MessageAgentUser.get_table_name()
    primary_key = "message_id"
    secondary_key = "agent_id"
    third_key = "user_id"

    from AFAAS.lib.message_user_agent import MessageUserAgent

    async def add(self, value: dict, id: str = MessageUserAgent.generate_uuid()) -> str:
        return await super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    async def update(self, message_id: str, agent_id: str, value: dict):
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return await super().update(key=key, value=value)

    async def delete(self, message_id: str, agent_id: str):
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return await super().delete(key=key)

    async def get(self, message_id: str, agent_id: str) -> MessageUserAgent:
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return await super().get(key=key)
