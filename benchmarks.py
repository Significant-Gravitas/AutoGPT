<<<<<<< HEAD
from pathlib import Path

from autogpt.agents import Agent
from autogpt.app.main import run_interaction_loop
from autogpt.commands import COMMAND_CATEGORIES
=======
import sys
from typing import Tuple

from autogpt.agent import Agent
>>>>>>> c29ec925 (WIP)
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

<<<<<<< HEAD
PROJECT_DIR = Path().resolve()

=======
def run_specific_agent(task) -> Tuple[str, int]:

    cycle_count = 0
>>>>>>> c29ec925 (WIP)

    agent = bootstrap_agent(task)
<<<<<<< HEAD
    run_interaction_loop(agent)


def bootstrap_agent(task):
    config = ConfigBuilder.build_config_from_env(workdir=PROJECT_DIR)
    config.continuous_mode = False
=======
    response = agent.start_interaction_loop()

    if response:
        cycle_count += 1



    return response, 1


def bootstrap_agent(task):
    config = ConfigBuilder.build_config_from_env()
    config.debug_mode = True
    config.continuous_mode = True
>>>>>>> c29ec925 (WIP)
    config.temperature = 0
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"
    Workspace.set_workspace_directory(config)
    Workspace.build_file_logger_path(config, config.workspace_path)
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )
    ai_config.command_registry = command_registry
    return Agent(
        memory=get_memory(config),
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=str(config.workspace_path),
    )


def get_command_registry(config: Config):
    command_registry = CommandRegistry()
    enabled_command_categories = [
        x for x in COMMAND_CATEGORIES if x not in config.disabled_command_categories
    ]
    for command_category in enabled_command_categories:
        command_registry.import_commands(command_category)
    return command_registry

if __name__ == "__main__":
    # The first argument is the script name itself, second is the task
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[1]
    run_specific_agent(task)
