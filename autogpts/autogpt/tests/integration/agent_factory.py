import pytest

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry


@pytest.fixture
def memory_json_file(config: Config):
    was_memory_backend = config.memory_backend

    config.memory_backend = "json_file"
    memory = get_memory(config)
    memory.clear()
    yield memory

    config.memory_backend = was_memory_backend


@pytest.fixture
def dummy_agent(config: Config, llm_provider, memory_json_file):
    command_registry = CommandRegistry()

    ai_config = AIConfig(
        ai_name="Dummy Agent",
        ai_role="Dummy Role",
        ai_goals=[
            "Dummy Task",
        ],
    )

    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_config=ai_config,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        memory=memory_json_file,
        legacy_config=config,
    )

    return agent
