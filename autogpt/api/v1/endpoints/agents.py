import uuid

from fastapi import APIRouter
from pydantic.main import BaseModel

router = APIRouter()


class WorkspaceConfigRequest(BaseModel):
    root: str


class WorkspaceRequest(BaseModel):
    configuration: WorkspaceConfigRequest


class AgnosticAgentRequest(BaseModel):
    workspace: WorkspaceRequest


class AgentRequest(AgnosticAgentRequest):
    pass


@router.post("/agents")
async def create_agents(body: AgentRequest):
    agent_id = uuid.uuid4().hex
    return {
        "agent_id": agent_id,
    }
