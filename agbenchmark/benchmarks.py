import sys
from typing import Tuple

from autogpt.agent import Agent
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.main import COMMAND_CATEGORIES
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

def run_specific_agent(task) -> Tuple[str, int]:

    cycle_count = 0

    agent = bootstrap_agent(task)
    response = agent.start_interaction_loop()

    if response:
        cycle_count += 1



    return response, 1


def bootstrap_agent(task):
    config = ConfigBuilder.build_config_from_env()
    config.debug_mode = True
    config.continuous_mode = True
    config.temperature = 0
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"
    workspace_directory = Workspace.get_workspace_directory(config)
    workspace_directory_path = Workspace.make_workspace(workspace_directory)
    Workspace.build_file_logger_path(config, workspace_directory_path)
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )
    ai_config.command_registry = command_registry
    system_prompt = ai_config.construct_full_prompt(config)
    return Agent(
        ai_name="Auto-GPT",
        memory=get_memory(config),
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=str(workspace_directory_path),
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
