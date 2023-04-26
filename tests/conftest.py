from pathlib import Path

import pytest

from autogpt.api_manager import ApiManager
from autogpt.api_manager import api_manager as api_manager_
from autogpt.config import Config
from autogpt.workspace import Workspace


@pytest.fixture()
def workspace_root(tmp_path) -> Path:
    return tmp_path / "home/users/monty/auto_gpt_workspace"


@pytest.fixture()
def workspace(workspace_root: Path) -> Workspace:
    workspace_root = Workspace.make_workspace(workspace_root)
    return Workspace(workspace_root, restrict_to_workspace=True)


@pytest.fixture()
def config(workspace: Workspace) -> Config:
    config = Config()

    # Do a little setup and teardown since the config object is a singleton
    old_ws_path = config.workspace_path
    config.workspace_path = workspace.root
    yield config
    config.workspace_path = old_ws_path


@pytest.fixture()
def api_manager() -> ApiManager:
    old_attrs = api_manager_.__dict__.copy()
    api_manager_.reset()
    yield api_manager_
    api_manager_.__dict__.update(old_attrs)
