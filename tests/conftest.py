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


def pytest_report_teststatus(report, config):
    if report.when == "call" and report.outcome == "skipped" and hasattr(report, "wasxfail"):
        reason = report.wasxfail
        if "benchmarks" in reason and os.environ.get("CI"):
            with open("xfail_reasons.log", "a") as f:
                f.write(f"Test: {report.nodeid}\nReason: {reason}\n\n")
