from pathlib import Path

import pytest
from dotenv import load_dotenv

from autogpt.workspace import Workspace

load_dotenv()


@pytest.fixture()
def workspace_root(tmp_path) -> Path:
    return tmp_path / "home/users/monty/auto_gpt_workspace"


@pytest.fixture()
def workspace(workspace_root: Path) -> Workspace:
    workspace_root = Workspace.make_workspace(workspace_root)
    return Workspace(workspace_root, restrict_to_workspace=True)
