import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import autogpt.memory.vector.memory_item as vector_memory_item
import autogpt.memory.vector.providers.base as memory_provider_base
from autogpt.config.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.memory.vector import get_memory
from autogpt.workspace import Workspace

pytest_plugins = ["tests.integration.agent_factory"]

PROXY = os.environ.get("PROXY")


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


@pytest.fixture
def memory_none(agent_test_config: Config):
    was_memory_backend = agent_test_config.memory_backend

    agent_test_config.set_memory_backend("no_memory")
    yield get_memory(agent_test_config, init=True)

    agent_test_config.set_memory_backend(was_memory_backend)


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture, embedding_dimension: int):
    mocker.patch.object(
        vector_memory_item,
        "get_embedding",
        return_value=[0.0255] * embedding_dimension,
    )
    mocker.patch.object(
        memory_provider_base,
        "get_embedding",
        return_value=[0.0255] * embedding_dimension,
    )


@pytest.fixture()
def api_manager() -> ApiManager:
    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    return ApiManager()
