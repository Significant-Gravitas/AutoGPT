import json
import os
import shutil
import sys
import time
from pathlib import Path  # noqa
from typing import Any, Dict, Generator

import pytest
from helicone.lock import HeliconeLockManager

from agbenchmark.reports.reports import (
    finalize_reports,
    generate_combined_suite_report,
    generate_single_call_report,
    session_finish,
    setup_dummy_dependencies,
)
from agbenchmark.start_benchmark import CONFIG_PATH, get_regression_data
from agbenchmark.utils.data_types import SuiteConfig


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
    parser.addoption("--category", action="store_true", default=False)
    parser.addoption("--nc", action="store_true", default=False)
    parser.addoption("--improve", action="store_true", default=False)
    parser.addoption("--maintain", action="store_true", default=False)
    parser.addoption("--test", action="store_true", default=None)
    parser.addoption("--no_dep", action="store_true", default=False)
    parser.addoption("--suite", action="store_true", default=False)


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


suite_reports: dict[str, list] = {}


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    challenge_data = item.funcargs.get("challenge_data", None)

    if not challenge_data:
        # this will only happen for dummy dependency setup tests
        return

    challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
    # this is a non same task suite, with the location pointing to a data.json
    is_suite = SuiteConfig.suite_data_if_suite(
        Path(__file__).parent.parent / Path(challenge_location)
    )

    try:
        # this is for a same_task suite pointing to the directory where the suite lives
        is_suite = SuiteConfig.deserialize(
            Path(__file__).parent.parent / Path(challenge_location) / "suite.json"
        )
    except Exception as e:
        pass

    flags = "--test" in sys.argv or "--maintain" in sys.argv or "--improve" in sys.argv

    if call.when == "call":
        test_name = ""
        # if it's a same task suite, we combine the report.
        # but not if it's a single --test
        if is_suite and is_suite.same_task and not flags:
            test_name = is_suite.prefix
            generate_combined_suite_report(item, challenge_data, challenge_location)
        else:
            # single non suite test
            test_name = challenge_data["name"]
            generate_single_call_report(item, call, challenge_data)
        # else: it's a same_task=false suite (tests aren't combined)
        HeliconeLockManager.write_custom_property("challenge", test_name)
    if call.when == "teardown":
        finalize_reports(item, challenge_data)

        # for separate task suites (same_task=false), their data is the same as a regular suite, but we combined the report at the end
        if is_suite and not is_suite.same_task and not flags:
            suite_reports.setdefault(is_suite.prefix, []).append(challenge_data["name"])


def pytest_sessionfinish(session: Any) -> None:
    """Called at the end of the session to save regression tests and info"""

    session_finish(suite_reports)


@pytest.fixture
def scores(request: Any) -> None:
    test_class_name = request.node.cls.__name__
    return request.node.cls.scores.get(test_class_name)


def pytest_generate_tests(metafunc: Any) -> None:
    """This is to generate the dummy dependencies each test class"""
    test_class_instance = metafunc.cls()

    if test_class_instance.setup_dependencies:
        test_class = metafunc.cls
        setup_dummy_dependencies(test_class_instance, test_class)
        setattr(test_class, "setup_dependencies", [])


# this is adding the dependency marker and category markers automatically from the json
def pytest_collection_modifyitems(items: Any, config: Any) -> None:
    data = get_regression_data()

    for item in items:
        # Assuming item.cls is your test class
        test_class_instance = item.cls()

        # if it's a dummy dependency setup test, we also skip
        if "test_method" not in item.name:
            continue

        # Then you can access your properties
        name = item.parent.cls.__name__
        dependencies = test_class_instance.data.dependencies

        # Filter dependencies if they exist in regression data if its an improvement test
        if (
            config.getoption("--improve")
            or config.getoption("--category")
            or test_class_instance.setup_dependencies  # same_task suite
        ):
            dependencies = [dep for dep in dependencies if not data.get(dep, None)]
        if (
            config.getoption("--test")
            or (  # separate task suite
                not test_class_instance.setup_dependencies
                and config.getoption("--suite")
            )
            or config.getoption("--no_dep")
            or config.getoption("--maintain")
        ):
            dependencies = []

        categories = test_class_instance.data.category

        # Add depends marker dynamically
        item.add_marker(pytest.mark.depends(on=dependencies, name=name))

        # Add category marker dynamically
        for category in categories:
            item.add_marker(getattr(pytest.mark, category))
