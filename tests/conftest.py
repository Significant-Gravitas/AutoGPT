import os
from pathlib import Path

import pytest

from AFAAS.core.agents.planner.main import PlannerAgent
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.tools.base import BaseToolsRegistry
from tests.dataset.agent_planner import agent_dataset


@pytest.fixture
def agent() -> PlannerAgent:
    return agent_dataset()


@pytest.fixture(scope="function", autouse=True)
def reset_environment_each_test():
    # Code to reset the environment before each test
    setup_environment()
    delete_logs()

    yield

    # Code to clean up after each test
    delete_logs()


def setup_environment():
    # Code to set up your environment for each test
    pass


def delete_logs():
    log_dir = Path(__file__).parent.parent / "logs"
    # Check if the directory exists
    if not log_dir.exists():
        print("Directory does not exist:", log_dir)
    else:
        # Iterate over all files in the directory
        for file in log_dir.iterdir():
            # Check if the file name starts with 'pytest_'
            if file.is_file() and file.name.startswith("pytest_"):
                print("Deleting:", file)
                try:
                    os.remove(file)
                except OSError as e:
                    print("Error while deleting file:", e)


@pytest.fixture
def local_workspace() -> AbstractFileWorkspace:
    return agent_dataset().workspace


@pytest.fixture
def empty_tool_registry() -> BaseToolsRegistry:
    registry = agent_dataset().tool_registry
    registry.tools = {}
    return registry
