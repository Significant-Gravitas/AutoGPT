from __future__ import annotations


import uuid
from typing import (
    TYPE_CHECKING,
)

from .base import BaseNoSQLTable

class MessagesUserAgentTable(BaseNoSQLTable):
    table_name = "messages_history"
    primary_key = "message_id"
    secondary_key = "user_id"
    third_key = "agent_id"