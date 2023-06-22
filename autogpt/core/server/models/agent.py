from typing import List

from pydantic import BaseModel


class AgentDefinition(BaseModel):
    data: str


class StartAgentResponse(BaseModel):
    name: str
    id: str
    session_id: str


class RunningAgent(BaseModel):
    name: str
    id: str
    session_id: str


class AgentListResponse(BaseModel):
    agents: List[RunningAgent]


class StopAgentReq(BaseModel):
    agent_id: str
    immediately: bool = False


class StopAgentResponse(BaseModel):
    agent_id: str
    status: str
