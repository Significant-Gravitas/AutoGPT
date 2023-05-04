import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from autogpt.config import Config
from autogpt.llm import ApiManager
from autogpt.workspace import Workspace

pytest_plugins = ["tests.integration.agent_factory"]


@pytest.fixture()
def workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "home/users/monty/auto_gpt_workspace"


@pytest.fixture()
def workspace(workspace_root: Path) -> Workspace:
    workspace_root = Workspace.make_workspace(workspace_root)
    return Workspace(workspace_root, restrict_to_workspace=True)


import pytest

from autogpt.config import Config


@pytest.fixture()
def config(request, mocker: MockerFixture, workspace: Workspace) -> Config:
    if hasattr(request, "param"):
        if "env_vars" in request.param:
            for key, value in request.param["env_vars"].items():
                mocker.patch.dict(os.environ, {key: value})
        if "mock_load_dotenv" in request.param:
            mocker.patch(
                "dotenv.load_dotenv", side_effect=request.param["mock_load_dotenv"]
            )

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
