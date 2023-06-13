import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from autogpt.agent.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import TypingConsoleHandler
from autogpt.memory.vector import get_memory
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


@pytest.fixture()
def config(mocker: MockerFixture, workspace: Workspace) -> Config:
    config = Config()

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


@pytest.fixture(autouse=True)
def patch_emit(monkeypatch):
    # convert plain_output to a boolean

    if bool(os.environ.get("PLAIN_OUTPUT")):

        def quick_emit(self, record: str):
            print(self.format(record))

        monkeypatch.setattr(TypingConsoleHandler, "emit", quick_emit)


@pytest.fixture
def agent(config: Config, workspace: Workspace) -> Agent:
    ai_config = AIConfig(
        ai_name="Base",
        ai_role="A base AI",
        ai_goals=[],
    )

    command_registry = CommandRegistry()
    ai_config.command_registry = command_registry

    config.set_memory_backend("json_file")
    memory_json_file = get_memory(config, init=True)

    system_prompt = ai_config.construct_full_prompt()

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
