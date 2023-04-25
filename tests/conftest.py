from pathlib import Path

import pytest
from dotenv import load_dotenv

from autogpt.api_manager import ApiManager
from autogpt.api_manager import api_manager as api_manager_
from autogpt.config import Config
from autogpt.workspace import Workspace
from tests.vcr.openai_filter import before_record_request

load_dotenv()


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


def pytest_addoption(parser):
    parser.addoption(
        "--use-api-key",
        action="store_true",
        help="Use api key for the tests that need an api key AND whose behavior changes with your code. This will save Open AI answers for future use.",
    )


def pytest_configure(config):
    pytest.use_api_key = config.getoption("--use-api-key")


@pytest.fixture
def vcr_config():
    return {
        "record_mode": "new_episodes",
    }


@pytest.fixture
def vcr_config():
    return {
        "record_mode": "new_episodes",
        "before_record_request": before_record_request,
        "filter_headers": [
            "authorization",
            "X-OpenAI-Client-User-Agent",
            "User-Agent",
        ],
    }
