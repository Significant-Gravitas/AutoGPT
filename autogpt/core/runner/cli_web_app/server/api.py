from pathlib import Path

from agent_protocol import StepHandler, StepResult
from colorama import Fore

from autogpt.agents import Agent
from autogpt.app.main import UserFeedback
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, ConfigBuilder
from autogpt.logs import logger
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

PROJECT_DIR = Path().resolve()


async def task_handler(task_input) -> StepHandler:
    task = task_input.__root__ if task_input else {}
    agent = bootstrap_agent(task.get("user_input"), False)

    next_command_name: str | None = None
    next_command_args: dict[str, str] | None = None

    async def step_handler(step_input) -> StepResult:
        step = step_input.__root__ if step_input else {}

        nonlocal next_command_name, next_command_args

        result = await interaction_step(
            agent,
            step.get("user_input"),
            step.get("user_feedback"),
            next_command_name,
            next_command_args,
        )

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
