import os
import shutil
from pathlib import Path

import pytest

from AFAAS.core.agents.planner.main import Plan, PlannerAgent
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.tools.base import AbstractToolRegistry
from AFAAS.lib.sdk.logger import AFAASLogger, logging
from AFAAS.lib.task.task import Task
from tests.dataset.agent_planner import agent_dataset
from tests.dataset.plan_familly_dinner import Task, _plan_familly_dinner, default_task

# LOG = AFAASLogger(name=__name__)
# LOG.setLevel(logging.ERROR)
os.environ["PYTEST_RUN"] = "true"

if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value

    # def pytest_exception_interact(node, call, report):
    #     print( call.excinfo.traceback[0] )
    #     pass


@pytest.fixture(scope="session")
def activate_integration_tests():
    # Use an environment variable to control the activation of integration tests

    return os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


# @pytest.fixture(scope='function', autouse=True)
# def capture_fixture_name(request):
#     yield
#     if hasattr(request.node, 'rep_setup'):
#         if request.node.parent.get_closest_marker("pytest.mark.asyncio"):
#             # This is an asynchronous test
#             fixture_name = request.node.function.__dict__.get('fixtureinfo', None)
#             if fixture_name:
#                 request.node.user_properties.append(("Fixture Used (Async)", fixture_name.name))
#         else:
#             # This is a synchronous test
#             fixture_name = request.node.function.__dict__.get('fixtureinfo', None)
#             if fixture_name:
#                 request.node.user_properties.append(("Fixture Used (Sync)", fixture_name.name))


@pytest.mark.asyncio
@pytest.fixture(scope="function")
async def agent() -> PlannerAgent:
    return await agent_dataset()


# # Higher-level fixture to intercept Plan fixtures
# @pytest.mark.asyncio
# @pytest.fixture(autouse=True)
# async def intercept_plan_fixtures(request):
#     # Identify fixtures that are used in the test and start with 'plan_'
#     plan_fixture_names = [name for name in request.fixturenames if name.startswith('plan_')]

#     for fixture_name in plan_fixture_names:
#         plan : Plan = request.getfixturevalue(fixture_name)
#         await plan.db_create()

# @pytest.mark.asyncio
# @pytest.fixture(autouse=True)
# def intercept_task_fixtures(request):
#     # Identify fixtures that are used in the test and start with 'plan_'
#     task_fixture_names = [name for name in request.fixturenames if name.startswith('task_')]

#     for fixture_name in task_fixture_names:
#         task : Task = request.getfixturevalue(fixture_name)

#         task.agent.plan.db_save()


@pytest.fixture(scope="function", autouse=True)
def reset_environment_each_test(default_task: Task):
    # AFAASLogger.setLevel(logging.ERROR)
    setup_environment()
    delete_logs(agent = default_task.agent)
    base_dir = Path("~/AFAAS/data/pytest").expanduser().resolve()
    print(base_dir)
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        shutil.rmtree(root)
        print(f"Deleted directory: {root}")

    yield

    # Code to clean up after each test
    delete_logs()


def setup_environment():
    # Code to set up your environment for each test
    pass


def delete_logs(agent : PlannerAgent = None):
    log_dir = agent.log_path if agent else Path("~/AFAAS/data/logs").expanduser().resolve()
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


@pytest.fixture(scope="function", autouse=True)
async def local_workspace() -> AbstractFileWorkspace:
    agent = await agent_dataset()
    return agent.workspace


@pytest.fixture(scope="function", autouse=True)
async def empty_tool_registry() -> AbstractToolRegistry:
    agent = await agent_dataset()
    registry = agent.tool_registry
    registry.tools_by_name = {}
    return registry


@pytest.fixture(scope="function", autouse=True)
async def local_workspace() -> AbstractFileWorkspace:
    agent = await agent_dataset()
    return agent.workspace


# In your pytest fixture
@pytest.fixture(scope="function", autouse=True)
def reset_singleton():
    Plan._instance = {}
    Plan.initialized = False
    yield
