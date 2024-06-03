from typing import Optional

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.config.config import Config
from forge.file_storage.base import FileStorage
from forge.llm.providers import MultiProvider

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings


def create_agent(
    agent_id: str,
    task: str,
    app_config: Config,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
) -> Agent:
    if not task:
        raise ValueError("No task specified for new agent")
    ai_profile = ai_profile or AIProfile()
    directives = directives or AIDirectives()

    agent = _configure_agent(
        agent_id=agent_id,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )

    return agent


def configure_agent_with_state(
    state: AgentSettings,
    app_config: Config,
    file_storage: FileStorage,
    llm_provider: MultiProvider,
) -> Agent:
    return _configure_agent(
        state=state,
        app_config=app_config,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )


def _configure_agent(
    app_config: Config,
    llm_provider: MultiProvider,
    file_storage: FileStorage,
    agent_id: str = "",
    task: str = "",
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
    state: Optional[AgentSettings] = None,
) -> Agent:
    if state:
        agent_state = state
    elif agent_id and task and ai_profile and directives:
        agent_state = state or create_agent_state(
            agent_id=agent_id,
            task=task,
            ai_profile=ai_profile,
            directives=directives,
            app_config=app_config,
        )
    else:
        raise TypeError(
            "Either (state) or (agent_id, task, ai_profile, directives)"
            " must be specified"
        )

    return Agent(
        settings=agent_state,
        llm_provider=llm_provider,
        file_storage=file_storage,
        legacy_config=app_config,
    )


def create_agent_state(
    agent_id: str,
    task: str,
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: Config,
) -> AgentSettings:
    return AgentSettings(
        agent_id=agent_id,
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )
