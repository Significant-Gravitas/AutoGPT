import json
import os
import shutil
import sys
import time
from pathlib import Path  # noqa
from typing import Any, Dict, Generator

import pytest

from agbenchmark.ReportManager import ReportManager
from agbenchmark.start_benchmark import (
    CONFIG_PATH,
    INFO_TESTS_PATH,
    REGRESSION_TESTS_PATH,
    get_regression_data,
)
from agbenchmark.utils import AGENT_NAME, calculate_success_percentage


def resolve_workspace(workspace: str) -> str:
    if workspace.startswith("${") and workspace.endswith("}"):
        # Extract the string inside ${...}
        path_expr = workspace[2:-1]

        # Check if it starts with "os.path.join"
        if path_expr.strip().startswith("os.path.join"):
            # Evaluate the path string
            path_value = eval(path_expr)

            # Replace the original string with the evaluated result
            return path_value
        else:
            raise ValueError("Invalid workspace path expression.")
    else:
        return os.path.abspath(Path(os.getcwd()) / workspace)


@pytest.fixture(scope="module")
def config(request: Any) -> None:
    print(f"Config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if isinstance(config["workspace"], str):
        config["workspace"] = resolve_workspace(config["workspace"])
    else:  # it's a input output dict
        config["workspace"]["input"] = resolve_workspace(config["workspace"]["input"])
        config["workspace"]["output"] = resolve_workspace(config["workspace"]["output"])

    return config


@pytest.fixture(autouse=True)
def workspace(config: Dict[str, Any]) -> Generator[str, None, None]:
    output_path = config["workspace"]

    # checks if its an input output paradigm
    if not isinstance(config["workspace"], str):
        output_path = config["workspace"]["output"]
        if not os.path.exists(config["workspace"]["input"]):
            os.makedirs(config["workspace"]["input"], exist_ok=True)

    # create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    yield config["workspace"]
    # teardown after test function completes

    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--mock", action="store_true", default=False)
    parser.addoption("--improve", action="store_true", default=False)
    parser.addoption("--maintain", action="store_true", default=False)
    parser.addoption("--test", action="store_true", default=None)


@pytest.fixture(autouse=True)
def check_regression(request: Any) -> None:
    test_name = request.node.parent.name
    data = get_regression_data()

    # Get the true location of the test
    challenge_location = getattr(request.node.parent.cls, "CHALLENGE_LOCATION", "")

    skip_string = f"Skipping {test_name} at {challenge_location}"

    # Check if the test name exists in the regression tests
    if request.config.getoption("--improve") and data.get(test_name, None):
        pytest.skip(f"{skip_string} because it's a regression test")
    elif request.config.getoption("--maintain") and not data.get(test_name, None):
        pytest.skip(f"{skip_string} because it's not a regression test")


# this is to get the challenge_data from every test
@pytest.fixture(autouse=True)
def challenge_data(request: Any) -> None:
    return request.param


@pytest.fixture(autouse=True, scope="session")
def mock(request: Any) -> None:
    return request.config.getoption("--mock")


@pytest.fixture(autouse=True, scope="function")
def timer(request: Any) -> Any:
    start_time = time.time()
    yield
    run_time = time.time() - start_time
    request.node.user_properties.append(("run_time", run_time))


# tests that consistently pass are considered regression tests
regression_manager = ReportManager(REGRESSION_TESTS_PATH)

# user facing reporting information
info_manager = ReportManager(INFO_TESTS_PATH)

INTERNAL_LOGS_PATH = Path(__file__).resolve().parent / "reports"

# internal db step in replacement track pass/fail rate
internal_info = ReportManager(str(INTERNAL_LOGS_PATH / "internal_info.json"))


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    challenge_data = item.funcargs.get("challenge_data", None)
    if call.when == "call":
        difficulty = (
            challenge_data["info"]["difficulty"] if challenge_data else "unknown"
        )
        dependencies = dependencies = (
            challenge_data["dependencies"] if challenge_data else []
        )
        # Extract the challenge_location from the class
        challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
        test_name = item.nodeid.split("::")[1]
        item.test_name = test_name

        test_details = {
            "difficulty": difficulty,
            "dependencies": dependencies,
            "data_path": challenge_location,
        }

        info_details: Any = {
            "data_path": challenge_location,
            "is_regression": False,
            "task": challenge_data["task"],
            "answer": challenge_data["ground"]["answer"],
            "description": challenge_data["info"]["description"],
            "metrics": {
                "difficulty": difficulty,
                "success": False,
            },
        }

        mock = "--mock" in sys.argv  # Check if --mock is in sys.argv

        if call.excinfo is None:
            info_details["metrics"]["success"] = True
        else:
            if not mock:  # don't remove if it's a mock test
                regression_manager.remove_test(test_name)
            info_details["metrics"]["fail_reason"] = str(call.excinfo.value)

        prev_test_results: list[bool]
        agent_tests: dict[str, list[bool]] = {}

        # if the structure is nested inside of the agent name
        if AGENT_NAME:
            agent_tests = internal_info.tests.get(AGENT_NAME, {})

        if agent_tests:
            prev_test_results = agent_tests.get(test_name, [])
        else:
            prev_test_results = internal_info.tests.get(test_name, [])

        if not mock:
            # only add if it's an actual test
            prev_test_results.append(info_details["metrics"]["success"])
            internal_info.add_test(test_name, prev_test_results, AGENT_NAME)

            # can calculate success rate regardless of mock
            info_details["metrics"]["success_%"] = calculate_success_percentage(
                prev_test_results
            )
        else:
            # can calculate success rate regardless of mock
            info_details["metrics"][
                "non_mock_success_%"
            ] = calculate_success_percentage(prev_test_results)

        if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
            # if the last 3 tests were successful, add to the regression tests
            info_details["is_regression"] = True
            regression_manager.add_test(test_name, test_details)

        # user facing reporting
        item.info_details = info_details
    if call.when == "teardown":
        run_time = dict(item.user_properties).get("run_time")

        info_details = getattr(item, "info_details", {})
        test_name = getattr(item, "test_name", "")

        if info_details and test_name:
            if run_time:
                info_details["metrics"][
                    "run_time"
                ] = f"{str(round(run_time, 3))} seconds"

                info_details["reached_cutoff"] = (
                    float(run_time) > challenge_data["cutoff"]
                )

            info_manager.add_test(test_name, info_details)


def pytest_sessionfinish(session: Any) -> None:
    """Called at the end of the session to save regression tests and info"""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    internal_info.save()
    info_manager.end_info_report(config)
    regression_manager.save()


# this is adding the dependency marker and category markers automatically from the json
def pytest_collection_modifyitems(items: Any, config: Any) -> None:
    data = get_regression_data()

    for item in items:
        # Assuming item.cls is your test class
        test_class_instance = item.cls()

        # Then you can access your properties
        name = item.parent.cls.__name__
        dependencies = test_class_instance.data.dependencies

        # Filter dependencies if they exist in regression data if its an improvement test
        if config.getoption("--improve"):
            dependencies = [dep for dep in dependencies if not data.get(dep, None)]
        elif config.getoption("--test"):
            dependencies = []

        categories = test_class_instance.data.category

        # Add depends marker dynamically
        item.add_marker(pytest.mark.depends(on=dependencies, name=name))

        # Add category marker dynamically
        for category in categories:
            item.add_marker(getattr(pytest.mark, category))
