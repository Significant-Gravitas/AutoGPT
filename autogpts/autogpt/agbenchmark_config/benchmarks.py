import asyncio
import sys
from pathlib import Path

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.main import _configure_openai_provider, run_interaction_loop
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, ConfigBuilder
from autogpt.logs.config import configure_logging
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.workspace import Workspace

PROJECT_DIR = Path().resolve()
LOG_DIR = Path(__file__).parent / "logs"


def run_specific_agent(task: str, continuous_mode: bool = False) -> None:
    agent = bootstrap_agent(task, continuous_mode)
    asyncio.run(run_interaction_loop(agent))


def bootstrap_agent(task: str, continuous_mode: bool) -> Agent:
    config = ConfigBuilder.build_config_from_env(workdir=PROJECT_DIR)
    config.debug_mode = False
    config.continuous_mode = continuous_mode
    config.continuous_limit = 20
    config.temperature = 0
    config.noninteractive_mode = True
    config.plain_output = True
    config.memory_backend = "no_memory"
    config.workspace_path = Workspace.init_workspace_directory(config)
    config.file_logger_path = Workspace.build_file_logger_path(config.workspace_path)

    configure_logging(config, LOG_DIR)

    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)

    ai_config = AIConfig(
        ai_name="AutoGPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
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

    return Agent(
        settings=agent_settings,
        llm_provider=_configure_openai_provider(config),
        command_registry=command_registry,
        memory=get_memory(config),
        legacy_config=config,
    )


if __name__ == "__main__":
    # The first argument is the script name itself, second is the task
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[1]
    run_specific_agent(task, continuous_mode=True)
