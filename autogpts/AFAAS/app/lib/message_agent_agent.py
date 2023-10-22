import enum
import uuid
from pydantic import BaseModel
from typing import Optional
from autogpts.autogpt.autogpt.core.configuration.schema import AFAASModel, AFAASMessageType


class MessageAgentAgent(AFAASModel) :
    message_id : str = 'MAA'+str(uuid.uuid4())
    message_type = AFAASMessageType.AGENT_AGENT.value
    agent_sender_id : str
    agent_receiver_id : str
    user_id : str
    message : str