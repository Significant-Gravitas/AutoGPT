from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from forge.config.ai_profile import AIProfile
from forge.file_storage.local import (
    FileStorage,
    FileStorageConfiguration,
    LocalFileStorage,
)
from forge.llm.providers import MultiProvider
from forge.logging.config import configure_logging

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.config import AppConfig, ConfigBuilder
from autogpt.app.main import _configure_llm_provider

pytest_plugins = [
    "tests.integration.agent_factory",
]


@pytest.fixture()
def tmp_project_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def app_data_dir(tmp_project_root: Path) -> Path:
    dir = tmp_project_root / "data"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


@pytest.fixture()
def storage(app_data_dir: Path) -> FileStorage:
    storage = LocalFileStorage(
        FileStorageConfiguration(root=app_data_dir, restrict_to_root=False)
    )
    storage.initialize()
    return storage


@pytest.fixture(scope="function")
def config(
    tmp_project_root: Path,
    app_data_dir: Path,
):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy"
    config = ConfigBuilder.build_config_from_env(project_root=tmp_project_root)

    config.app_data_dir = app_data_dir

    config.noninteractive_mode = True

    yield config


@pytest.fixture(scope="session")
def setup_logger():
    configure_logging(
        debug=True,
        log_dir=Path(__file__).parent / "logs",
        plain_console_output=True,
    )


@pytest.fixture
def llm_provider(config: AppConfig) -> MultiProvider:
    return _configure_llm_provider(config)


@pytest.fixture
def agent(
    config: AppConfig, llm_provider: MultiProvider, storage: FileStorage
) -> Agent:
    ai_profile = AIProfile(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        agent_id=f"AutoGPT-test-agent-{str(uuid.uuid4())[:8]}",
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            allow_fs_access=not config.restrict_to_workspace,
            use_functions_api=config.openai_functions,
        ),
        history=Agent.default_settings.history.model_copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        file_storage=storage,
        app_config=config,
    )
    return agent
