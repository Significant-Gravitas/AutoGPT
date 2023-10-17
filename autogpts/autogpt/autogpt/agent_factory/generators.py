from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.config import Config
    from autogpt.core.resource.model_providers.schema import ChatModelProvider

from autogpt.config.ai_directives import AIDirectives

from .configurators import _configure_agent
from .profile_generator import generate_agent_profile_for_task


async def generate_agent_for_task(
    task: str,
    app_config: "Config",
    llm_provider: "ChatModelProvider",
) -> "Agent":
    base_directives = AIDirectives.from_file(app_config.prompt_settings_file)
    ai_profile, task_directives = await generate_agent_profile_for_task(
        task=task,
        app_config=app_config,
        llm_provider=llm_provider,
    )
    return _configure_agent(
        task=task,
        ai_profile=ai_profile,
        directives=base_directives + task_directives,
        app_config=app_config,
        llm_provider=llm_provider,
    )
