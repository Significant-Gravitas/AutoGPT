import pytest
from forge.config.ai_profile import AIProfile
from forge.config.config import Config
from forge.file_storage import FileStorageBackendName, get_storage

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings


@pytest.fixture
def dummy_agent(config: Config, llm_provider, memory_json_file):
    ai_profile = AIProfile(
        ai_name="Dummy Agent",
        ai_role="Dummy Role",
        ai_goals=[
            "Dummy Task",
        ],
    )

    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            use_functions_api=config.openai_functions,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )

    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend, root_path="data", restrict_to_root=restrict_to_root
    )
    file_storage.initialize()

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        file_storage=file_storage,
        legacy_config=config,
    )

    return agent
