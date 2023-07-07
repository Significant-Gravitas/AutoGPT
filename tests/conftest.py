import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from pytest_mock import MockerFixture

from autogpt.agent.agent import Agent
from autogpt.config import AIConfig, Config, ConfigBuilder
from autogpt.config.ai_config import AIConfig
from autogpt.llm.api_manager import ApiManager
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
def config(
    temp_plugins_config_file: str, mocker: MockerFixture, workspace: Workspace
) -> Config:
    config = ConfigBuilder.build_config_from_env()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy"

    config.plugins_dir = "tests/unit/data/test_plugins"
    config.plugins_config_file = temp_plugins_config_file

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
        file_logger_path=workspace.get_path("file_logger.txt"),
    )
    yield config


@pytest.fixture()
def api_manager() -> ApiManager:
    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    return ApiManager()


@pytest.fixture
def agent(config: Config, workspace: Workspace) -> Agent:
    ai_config = AIConfig(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    command_registry = CommandRegistry()
    ai_config.command_registry = command_registry
    config.memory_backend = "json_file"
    memory_json_file = get_memory(config)
    memory_json_file.clear()

    system_prompt = ai_config.construct_full_prompt(config)

    return Agent(
        ai_name=ai_config.ai_name,
        memory=memory_json_file,
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )
