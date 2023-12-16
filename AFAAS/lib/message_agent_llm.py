import enum
import uuid
from typing import Optional

from pydantic import BaseModel

from AFAAS.configs.schema  import (
    AFAASMessageType, AFAASModel)
from AFAAS.lib.utils.json_schema import JSONSchema


class MessageAgentLLM(AFAASModel):
    message_id: str = "MAL" + str(uuid.uuid4())
    message_type = AFAASMessageType.AGENT_LLM.value
    agent_sender_id: str
    agent_receiver_id: str
    user_id: str
    message: str
    function: JSONSchema
    process_context: str
    task_context: str
