import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agents import Agent
from autogpt.app.main import _configure_openai_provider
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.core.resource.model_providers import ChatModelProvider, OpenAIProvider
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.config import configure_logging
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace

pytest_plugins = [
    "tests.integration.agent_factory",
    "tests.integration.memory.utils",
    "tests.vcr",
]


@pytest.fixture()
def workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "home/users/monty/auto_gpt_workspace"


@pytest.fixture()
def workspace(workspace_root: Path) -> Workspace:
    workspace_root = Workspace.make_workspace(workspace_root)
    return Workspace(workspace_root, restrict_to_workspace=True)


@pytest.fixture
def temp_plugins_config_file():
    """Create a plugins_config.yaml file in a temp directory so that it doesn't mess with existing ones"""
    config_directory = TemporaryDirectory()
    config_file = os.path.join(config_directory.name, "plugins_config.yaml")
    with open(config_file, "w+") as f:
        f.write(yaml.dump({}))

    yield config_file


@pytest.fixture()
def config(temp_plugins_config_file: str, mocker: MockerFixture, workspace: Workspace):
    config = ConfigBuilder.build_config_from_env(workspace.root.parent)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy"

    config.workspace_path = workspace.root

    config.plugins_dir = "tests/unit/data/test_plugins"
    config.plugins_config_file = temp_plugins_config_file

    config.noninteractive_mode = True
    config.plain_output = True

    # avoid circular dependency
    from autogpt.plugins.plugins_config import PluginsConfig

    config.plugins_config = PluginsConfig.load_config(
        plugins_config_file=config.plugins_config_file,
        plugins_denylist=config.plugins_denylist,
        plugins_allowlist=config.plugins_allowlist,
    )

    # Do a little setup and teardown since the config object is a singleton
    mocker.patch.multiple(
        config,
        workspace_path=workspace.root,
        file_logger_path=workspace.get_path("file_logger.log"),
    )
    yield config


@pytest.fixture(scope="session")
def setup_logger(config: Config):
    configure_logging(config, Path(__file__).parent / "logs")


@pytest.fixture()
def api_manager() -> ApiManager:
    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    return ApiManager()


@pytest.fixture
def llm_provider(config: Config) -> OpenAIProvider:
    return _configure_openai_provider(config)


@pytest.fixture
def agent(config: Config, llm_provider: ChatModelProvider) -> Agent:
    ai_config = AIConfig(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    command_registry = CommandRegistry()
    config.memory_backend = "json_file"
    memory_json_file = get_memory(config)
    memory_json_file.clear()

    return Agent(
        memory=memory_json_file,
        llm_provider=llm_provider,
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
    )
