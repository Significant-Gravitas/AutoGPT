import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.main import _configure_openai_provider
from autogpt.config import AIProfile, Config, ConfigBuilder
from autogpt.core.resource.model_providers import ChatModelProvider, OpenAIProvider
from autogpt.file_workspace.local import (
    FileWorkspace,
    FileWorkspaceConfiguration,
    LocalFileWorkspace,
)
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.config import configure_logging
from autogpt.models.command_registry import CommandRegistry

pytest_plugins = [
    "tests.integration.agent_factory",
    "tests.integration.memory.utils",
    "tests.vcr",
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
def agent_data_dir(app_data_dir: Path) -> Path:
    return app_data_dir / "agents/AutoGPT"


@pytest.fixture()
def workspace_root(agent_data_dir: Path) -> Path:
    return agent_data_dir / "workspace"


@pytest.fixture()
def workspace(workspace_root: Path) -> FileWorkspace:
    workspace = LocalFileWorkspace(FileWorkspaceConfiguration(root=workspace_root))
    workspace.initialize()
    return workspace


@pytest.fixture
def temp_plugins_config_file():
    """
    Create a plugins_config.yaml file in a temp directory
    so that it doesn't mess with existing ones.
    """
    config_directory = TemporaryDirectory()
    config_file = Path(config_directory.name) / "plugins_config.yaml"
    with open(config_file, "w+") as f:
        f.write(yaml.dump({}))

    yield config_file


@pytest.fixture(scope="function")
def config(
    temp_plugins_config_file: Path,
    tmp_project_root: Path,
    app_data_dir: Path,
    mocker: MockerFixture,
):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy"
    config = ConfigBuilder.build_config_from_env(project_root=tmp_project_root)

    config.app_data_dir = app_data_dir

    config.plugins_dir = "tests/unit/data/test_plugins"
    config.plugins_config_file = temp_plugins_config_file

    config.logging.log_dir = Path(__file__).parent / "logs"
    config.logging.plain_console_output = True
    config.noninteractive_mode = True

    # avoid circular dependency
    from autogpt.plugins.plugins_config import PluginsConfig

    config.plugins_config = PluginsConfig.load_config(
        plugins_config_file=config.plugins_config_file,
        plugins_denylist=config.plugins_denylist,
        plugins_allowlist=config.plugins_allowlist,
    )
    yield config


@pytest.fixture(scope="session")
def setup_logger(config: Config):
    configure_logging(**config.logging.dict())


@pytest.fixture()
def api_manager() -> ApiManager:
    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    return ApiManager()


@pytest.fixture
def llm_provider(config: Config) -> OpenAIProvider:
    return _configure_openai_provider(config)


@pytest.fixture
def agent(
    agent_data_dir: Path, config: Config, llm_provider: ChatModelProvider
) -> Agent:
    ai_profile = AIProfile(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    command_registry = CommandRegistry()

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = config.openai_functions

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
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=config,
    )
    agent.attach_fs(agent_data_dir)
    return agent
