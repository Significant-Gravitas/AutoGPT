from typing import Optional

from autogpt.agent_manager import AgentManager
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.logs.config import configure_chat_plugins
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins


def create_agent(
    task: str,
    ai_profile: AIProfile,
    app_config: Config,
    llm_provider: ChatModelProvider,
    directives: Optional[AIDirectives] = None,
) -> Agent:
    if not task:
        raise ValueError("No task specified for new agent")
    if not directives:
        directives = AIDirectives.from_file(app_config.prompt_settings_file)

    agent = _configure_agent(
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
        llm_provider=llm_provider,
    )

    agent.state.agent_id = AgentManager.generate_id(agent.ai_profile.ai_name)

    return agent


def configure_agent_with_state(
    state: AgentSettings,
    app_config: Config,
    llm_provider: ChatModelProvider,
) -> Agent:
    return _configure_agent(
        state=state,
        app_config=app_config,
        llm_provider=llm_provider,
    )


def _configure_agent(
    app_config: Config,
    llm_provider: ChatModelProvider,
    task: str = "",
    ai_profile: Optional[AIProfile] = None,
    directives: Optional[AIDirectives] = None,
    state: Optional[AgentSettings] = None,
) -> Agent:
    if not (state or task and ai_profile and directives):
        raise TypeError(
            "Either (state) or (task, ai_profile, directives) must be specified"
        )

    app_config.plugins = scan_plugins(app_config)
    configure_chat_plugins(app_config)

    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry.with_command_modules(
        modules=COMMAND_CATEGORIES,
        config=app_config,
    )

    agent_state = state or create_agent_state(
        task=task,
        ai_profile=ai_profile,
        directives=directives,
        app_config=app_config,
    )

    # TODO: configure memory

    return Agent(
        settings=agent_state,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=app_config,
    )


def create_agent_state(
    task: str,
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: Config,
) -> AgentSettings:
    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = app_config.openai_functions

    return AgentSettings(
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
            plugins=app_config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )
