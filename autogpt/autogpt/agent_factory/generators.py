from __future__ import annotations

from typing import TYPE_CHECKING

from forge.file_storage.base import FileStorage

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.app.config import AppConfig
    from forge.llm.providers import MultiProvider

from .configurators import _configure_agent
from .profile_generator import generate_agent_profile_for_task


async def generate_agent_for_task(
    agent_id: str,
    task: str,
    app_config: AppConfig,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
) -> Agent:
    ai_profile, task_directives = await generate_agent_profile_for_task(
        task=task,
        app_config=app_config,
        llm_provider=llm_provider,
    )
    return _configure_agent(
        agent_id=agent_id,
        task=task,
        ai_profile=ai_profile,
        directives=task_directives,
        app_config=app_config,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )
