from __future__ import annotations

from .base import BaseNoSQLTable


class MessagesUserAgentTable(BaseNoSQLTable):
    table_name = "messages_history"
    primary_key = "message_id"
    secondary_key = "agent_id"
    third_key = "user_id"


    from AFAAS.lib.message_agent_user import MessageAgentUser

    def add(self, value: dict, id: str = MessageAgentUser.generate_uuid()) -> str:
        return super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    def update(self, message_id: str, agent_id: str, value: dict):
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return super().update(key=key, value=value)

    def delete(self, message_id: str, agent_id: str):
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return super().delete(key=key)

    def get(self, message_id: str, agent_id: str) -> MessageAgentUser:
        key = MessagesUserAgentTable.Key(
            primary_key=str(message_id),
            secondary_key=str(agent_id),
        )
        return super().get(key=key)

