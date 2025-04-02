from typing import Dict, Any, Optional, TypedDict
from pydantic import BaseModel
from enum import Enum

class EventType(str, Enum):
    RECORD_FLAGGED = "record.flagged"
    RECORD_COMPLIANT = "record.compliant"
    RECORD_UNFLAGGED = "record.unflagged"
    USER_SUSPENDED = "user.suspended"
    USER_UNSUSPENDED = "user.unsuspended"
    USER_BANNED = "user.banned"
    USER_UNBANNED = "user.unbanned"
    USER_COMPLIANT = "user.compliant"

class IffyWebhookEvent(BaseModel):
    event: EventType
    payload: Dict[str, Any]
    timestamp: str

class UserData(TypedDict):
    clientId: str
    email: Optional[str]
    name: Optional[str]
    username: Optional[str]

class IffyPayload(BaseModel):
    clientId: str
    name: str
    entity: str = "block_execution"
    metadata: Dict[str, Any]
    content: Dict[str, Any]
    user: Dict[str, Any]

class ModerationResult(BaseModel):
    is_safe: bool
    reason: str 

class BlockContentForModeration(BaseModel):
    graph_id: str
    graph_exec_id: str
    node_id: str
    block_id: str
    block_name: str
    block_type: str
    input_data: Dict[str, Any] 
