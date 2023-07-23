from pathlib import Path

from agent_protocol import Agent as AgentProtocol
from agent_protocol import StepHandler, StepResult
from colorama import Fore

from autogpt.agents import Agent
from autogpt.app.main import UserFeedback
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.logs import logger
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

PROJECT_DIR = Path().resolve()


async def task_handler(task_input) -> StepHandler:
    agent = bootstrap_agent(task_input)

    next_command_name: str | None
    next_command_args: dict[str, str] | None

    async def step_handler(step_input) -> StepResult:
        result = await interaction_step(
            agent,
            step_input["user_input"],
            step_input["user_feedback"],
            next_command_name,
            next_command_args,
        )

        nonlocal next_command_name, next_command_args
        next_command_name = result["next_step_command_name"] if result else None
        next_command_args = result["next_step_command_args"] if result else None

        if not result:
            return StepResult(output=None, is_last=True)
        return StepResult(output=result)

    return step_handler


async def interaction_step(
    agent: Agent,
    user_input,
    user_feedback: UserFeedback | None,
    command_name: str | None,
    command_args: dict[str, str] | None,
):
    """Run one step of the interaction loop."""
    if user_feedback == UserFeedback.EXIT:
        return
    if user_feedback == UserFeedback.TEXT:
        command_name = "human_feedback"

    result: str | None = None

    if command_name is not None:
        result = agent.execute(command_name, command_args, user_input)
        if result is None:
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
            return

    next_command_name, next_command_args, assistant_reply_dict = agent.think()

    return {
        "config": agent.config,
        "ai_config": agent.ai_config,
        "result": result,
        "assistant_reply_dict": assistant_reply_dict,
        "next_step_command_name": next_command_name,
        "next_step_command_args": next_command_args,
    }


def bootstrap_agent(task):
    config = ConfigBuilder.build_config_from_env(workdir=PROJECT_DIR)
    config.continuous_mode = False
    config.temperature = 0
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"
    Workspace.set_workspace_directory(config)
    Workspace.build_file_logger_path(config, config.workspace_path)
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task.user_input],
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


AgentProtocol.handle_task(task_handler)
