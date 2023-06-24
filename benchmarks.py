from autogpt.agent import Agent
from autogpt.config import AIConfig, Config
from autogpt.memory.vector import get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace
from tests.integration.agent_factory import get_command_registry


def run_task(task) -> None:
    agent = bootstrap_agent(task)
    agent.start_interaction_loop()


def bootstrap_agent(task):
    config = Config()
    config.set_continuous_mode(False)
    config.set_temperature(0)
    config.plain_output = True
    command_registry = get_command_registry(config)
    config.memory_backend = "no_memory"
    workspace_directory = Workspace.get_workspace_directory(config)
    workspace_directory_path = Workspace.make_workspace(workspace_directory)
    Workspace.build_file_logger_path(config, workspace_directory_path)
    ai_config = AIConfig(
        ai_name="Auto-GPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task.user_input],
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
