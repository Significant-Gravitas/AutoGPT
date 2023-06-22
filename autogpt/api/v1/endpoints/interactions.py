import uuid
from pathlib import Path

from fastapi import APIRouter
from pydantic.main import BaseModel

from autogpt.agent import Agent
from autogpt.api.v1.endpoints.agents import WorkspaceRequest
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace
from tests.integration.agent_factory import get_command_registry

router = APIRouter()

class InteractionRequest(BaseModel):
    workspace: WorkspaceRequest
    user_input: str
@router.post("/agents/{agent_id}/interactions")
async def create_interactions(agent_id: str, body: InteractionRequest):
    config = Config()
    config.set_continuous_mode(False)
    config.set_temperature(0)
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"
    Workspace.build_file_logger_path(config, Path(body.workspace.configuration.root))
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[body.user_input],
    )

    ai_config.command_registry = command_registry
    system_prompt = ai_config.construct_full_prompt(config)

    agent = Agent(
        ai_name="File System Agent",
        memory=get_memory(config),
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=body.workspace.configuration.root,
    )

    agent.start_interaction_loop()
