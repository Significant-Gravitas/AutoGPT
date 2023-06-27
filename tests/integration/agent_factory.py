import pytest

from autogpt.agent import Agent
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.workspace import Workspace


@pytest.fixture
def memory_json_file(config: Config):
    was_memory_backend = config.memory_backend

    config.memory_backend = "json_file"
    memory = get_memory(config)
    memory.clear()
    yield memory

    config.memory_backend = was_memory_backend


@pytest.fixture
def dummy_agent(config: Config, memory_json_file, workspace: Workspace):
    command_registry = CommandRegistry()

    ai_config = AIConfig(
        ai_name="Dummy Agent",
        ai_role="Dummy Role",
        ai_goals=[
            "Dummy Task",
        ],
    )
    ai_config.command_registry = command_registry

    agent = Agent(
        ai_name="Dummy Agent",
        memory=memory_json_file,
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        next_action_count=0,
        system_prompt="dummy_prompt",
        triggering_prompt="dummy triggering prompt",
        workspace_directory=workspace.root,
    )

    return agent
