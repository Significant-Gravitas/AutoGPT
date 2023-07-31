import os
import sys
from pathlib import Path
from typing import Tuple

from autogpt.agents import Agent
from autogpt.app.main import run_interaction_loop
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

PROJECT_DIR = Path().resolve()


def run_specific_agent(task, continuous_mode=False) -> Tuple[str, int]:
    agent = bootstrap_agent(task, continuous_mode)
    run_interaction_loop(agent)


def bootstrap_agent(task, continuous_mode) -> Agent:
    config = ConfigBuilder.build_config_from_env(workdir=PROJECT_DIR)
    config.debug_mode = True
    config.continuous_mode = continuous_mode
    config.temperature = 0
    config.plain_output = True
    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)
    config.memory_backend = "no_memory"
    config.workspace_path = Workspace.init_workspace_directory(config)
    config.file_logger_path = Workspace.build_file_logger_path(config.workspace_path)
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
    )


if __name__ == "__main__":
    # The first argument is the script name itself, second is the task
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[1]
    run_specific_agent(task, continuous_mode=True)
