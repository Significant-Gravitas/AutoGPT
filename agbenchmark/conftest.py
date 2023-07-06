import json
import os
import shutil
from pathlib import Path  # noqa
from typing import Any, Dict, Generator, List

import pytest

from agbenchmark.start_benchmark import CONFIG_PATH, REGRESSION_TESTS_PATH
from agbenchmark.tests.regression.RegressionManager import RegressionManager


def get_dynamic_workspace(config: Dict[str, Any]) -> str:
    # Extract the string inside ${...}
    path_expr = config["workspace"][2:-1]

    # Check if it starts with "os.path.join"
    if path_expr.strip().startswith("os.path.join"):
        # Evaluate the path string
        path_value = eval(path_expr)

        # Replace the original string with the evaluated result
        return path_value
    else:
        raise ValueError("Invalid workspace path expression.")


@pytest.fixture(scope="module")
def config(request: Any) -> None:
    print(f"Config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if config.get("workspace", "").startswith("${") and config.get(
        "workspace", ""
    ).endswith("}"):
        path = get_dynamic_workspace(config)
        config["workspace"] = path
    else:
        config["workspace"] = Path(os.getcwd()) / config["workspace"]
    return config


@pytest.fixture(scope="module", autouse=True)
def workspace(config: Dict[str, Any]) -> Generator[str, None, None]:
    yield config["workspace"]
    # teardown after test function completes
    for filename in os.listdir(config["workspace"]):
        file_path = os.path.join(config["workspace"], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--mock", action="store_true", default=False)


regression_manager = RegressionManager(REGRESSION_TESTS_PATH)


# this is to get the challenge_data from every test
@pytest.fixture(autouse=True)
def challenge_data(request: Any) -> None:
    return request.param


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    if call.when == "call":
        challenge_data = item.funcargs.get("challenge_data", None)
        difficulty = challenge_data.info.difficulty if challenge_data else "unknown"
        dependencies = challenge_data.dependencies if challenge_data else []
        parts = item.nodeid.split("::")[0].split("/")
        agbenchmark_index = parts.index("agbenchmark")
        file_path = "/".join(parts[agbenchmark_index:])
        test_details = {
            "difficulty": difficulty,
            "dependencies": dependencies,
            "test": file_path,
        }

        print("pytest_runtest_makereport", test_details)
        if call.excinfo is None:
            regression_manager.add_test(item.nodeid.split("::")[1], test_details)
        else:
            regression_manager.remove_test(item.nodeid.split("::")[1])


def pytest_collection_modifyitems(items: List[Any]) -> None:
    """Called once all test items are collected. Used
    to add regression and depends markers to collected test items."""
    for item in items:
        # regression add
        if item.nodeid.split("::")[1] in regression_manager.tests:
            print(regression_manager.tests)
            item.add_marker(pytest.mark.regression)


def pytest_sessionfinish() -> None:
    """Called at the end of the session to save regression tests"""
    regression_manager.save()


# this is so that all tests can inherit from the Challenge class
def pytest_generate_tests(metafunc: Any) -> None:
    if "challenge_data" in metafunc.fixturenames:
        # Get the instance of the test class
        test_class = metafunc.cls()

        # Generate the parameters
        params = test_class.data

        # Add the parameters to the test function
        metafunc.parametrize("challenge_data", [params], indirect=True)
