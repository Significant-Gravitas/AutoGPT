from pathlib import Path

from fastapi import APIRouter
from pydantic.main import BaseModel

from autogpt.agent import Agent
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from tests.integration.agent_factory import get_command_registry

router = APIRouter()


class Task(BaseModel):
    content: str
    workspace_location: str


@router.post("/tasks")
async def tasks(body: Task):
    config = Config()
    config.set_continuous_mode(True)
    config.set_temperature(0)
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"

    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[body.content],
    )
    workspace_directory = Path(body.workspace_location)
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
        workspace_directory=workspace_directory,
    )
    agent.start_interaction_loop()
