import enum
import uuid
from pydantic import BaseModel
from typing import Optional
from autogpts.autogpt.autogpt.core.configuration.schema import SystemSettings, AFAASMessageType
from autogpts.autogpt.autogpt.core.utils.json_schema import JSONSchema


class MessageAgentLLM(SystemSettings) :
    message_id : str = 'MAL'+str(uuid.uuid4())
    message_type = AFAASMessageType.AGENT_LLM.value
    agent_sender_id : str
    agent_receiver_id : str
    user_id : str
    message : str
    function : JSONSchema
    process_context : str 
    task_context : str 