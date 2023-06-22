import uuid
from pathlib import Path

from fastapi import APIRouter
from pydantic.main import BaseModel

from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace
from tests.integration.agent_factory import get_command_registry

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
        "workspace": body.workspace,
    }
