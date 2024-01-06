import os
from pathlib import Path

import pytest

from AFAAS.core.agents.planner.main import PlannerAgent
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.tools.base import BaseToolsRegistry
from tests.dataset.agent_planner import agent_dataset


@pytest.fixture(scope="session")
def activate_integration_tests():
    # Use an environment variable to control the activation of integration tests
    return os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

@pytest.fixture
def agent() -> PlannerAgent:
    return agent_dataset()

#FIXME: Issue #99 https://github.com/ph-ausseil/afaas/issues/99 is a prerequisite for this
# # Higher-level fixture to intercept Plan fixtures
# @pytest.fixture(autouse=True)
# def intercept_plan_fixtures(request):
#     # Identify fixtures that are used in the test and start with 'plan_'
#     plan_fixture_names = [name for name in request.fixturenames if name.startswith('plan_')]

#     for fixture_name in plan_fixture_names:
#         plan = request.getfixturevalue(fixture_name)
#         plan.agent.create_in_db()
#         plan.create_in_db(agent = plan.agent)

# @pytest.fixture(autouse=True)
# def intercept_plan_fixtures(request):
#     # Identify fixtures that are used in the test and start with 'plan_'
#     task_fixture_names = [name for name in request.fixturenames if name.startswith('task_')]

#     for fixture_name in task_fixture_names:
#         task = request.getfixturevalue(fixture_name)
#         task.agent.create_in_db()
#         task.agent.plan.create_in_db(agent = task.agent)

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
