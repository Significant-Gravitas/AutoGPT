import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Generator

import pytest

from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH
from agbenchmark.config import load_agbenchmark_config
from agbenchmark.reports.reports import (
    finalize_reports,
    generate_single_call_report,
    session_finish,
)
from agbenchmark.utils.challenge import Challenge

GLOBAL_TIMEOUT = (
    1500  # The tests will stop after 25 minutes so we can send the reports.
)

logger = logging.getLogger(__name__)

pytest_plugins = ["agbenchmark.utils.dependencies"]
collect_ignore = ["challenges"]
suite_reports: dict[str, list] = {}


@pytest.fixture(scope="module")
def config() -> dict:
    """
    Loads the applicable benchmark configuration and returns a config `dict`.

    Returns:
        dict: A dictionary with the benchmark config at `config["AgentBenchmarkConfig"]`
    """
    config = {}

    config["AgentBenchmarkConfig"] = load_agbenchmark_config()

    return config


@pytest.fixture(autouse=True)
def temp_folder() -> Generator[Path, None, None]:
    """
    Pytest fixture that sets up and tears down the temporary folder for each test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    """

    # create output directory if it doesn't exist
    if not os.path.exists(TEMP_FOLDER_ABS_PATH):
        os.makedirs(TEMP_FOLDER_ABS_PATH, exist_ok=True)

    yield TEMP_FOLDER_ABS_PATH
    # teardown after test function completes
    if not os.getenv("KEEP_TEMP_FOLDER_FILES"):
        for filename in os.listdir(TEMP_FOLDER_ABS_PATH):
            file_path = os.path.join(TEMP_FOLDER_ABS_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Pytest hook that adds command-line options to the `pytest` command.
    The added options are specific to agbenchmark and control its behavior:
    * `--mock` is used to run the tests in mock mode.
    * `--host` is used to specify the host for the tests.
    * `--category` is used to run only tests of a specific category.
    * `--nc` is used to run the tests without caching.
    * `--cutoff` is used to specify a cutoff time for the tests.
    * `--improve` is used to run only the tests that are marked for improvement.
    * `--maintain` is used to run only the tests that are marked for maintenance.
    * `--explore` is used to run the tests in exploration mode.
    * `--test` is used to run a specific test.
    * `--no_dep` is used to run the tests without dependencies.
    * `--keep_answers` is used to keep the answers of the tests.

    Args:
        parser: The Pytest CLI parser to which the command-line options are added.
    """
    parser.addoption("--no_dep", action="store_true")
    parser.addoption("--mock", action="store_true")
    parser.addoption("--host", default=None)
    parser.addoption("--nc", action="store_true")
    parser.addoption("--cutoff", action="store_true")
    parser.addoption("--category", action="append")
    parser.addoption("--test", default=None)
    parser.addoption("--improve", action="store_true")
    parser.addoption("--maintain", action="store_true")
    parser.addoption("--explore", action="store_true")
    parser.addoption("--keep-answers", action="store_true")


@pytest.fixture(autouse=True)
def check_regression(request: pytest.FixtureRequest) -> None:
    """
    Fixture that checks for every test if it should be treated as a regression test,
    and whether to skip it based on that.

    The test name is retrieved from the `request` object. Regression reports are loaded
    from the path specified in the benchmark configuration.

    Effect:
    * If the `--improve` option is used and the current test is considered a regression
      test, it is skipped.
    * If the `--maintain` option is used and the current test  is not considered a
      regression test, it is also skipped.

    Args:
        request: The request object from which the test name and the benchmark
            configuration are retrieved.
    """
    test_name = request.node.parent.name
    agent_benchmark_config = load_agbenchmark_config()
    with contextlib.suppress(Exception):
        test = agent_benchmark_config.get_regression_reports_path()
        data = json.loads(test)
        challenge_location = getattr(request.node.parent.cls, "CHALLENGE_LOCATION", "")

        skip_string = f"Skipping {test_name} at {challenge_location}"

        # Check if the test name exists in the regression tests
        if request.config.getoption("--improve") and data.get(test_name, None):
            pytest.skip(f"{skip_string} because it's a regression test")
        elif request.config.getoption("--maintain") and not data.get(test_name, None):
            pytest.skip(f"{skip_string} because it's not a regression test")


# this is to get the challenge_data from every test
@pytest.fixture(autouse=True)
def challenge_data(request: pytest.FixtureRequest) -> Any:
    """
    Pytest fixture that provides the challenge data for each test.
    The challenge data is retrieved from the request object's parameters.

    Args:
        request: The pytest request object, which should contain challenge data.
    """
    return request.param


@pytest.fixture(autouse=True, scope="session")
def mock(request: pytest.FixtureRequest) -> bool:
    """
    Pytest fixture that retrieves the value of the `--mock` command-line option.
    The `--mock` option is used to run the tests in mock mode.

    Args:
        request: The `pytest.FixtureRequest` from which the `--mock` option value
            is retrieved.

    Returns:
        bool: Whether `--mock` is set for this session.
    """
    return request.config.getoption("--mock")


@pytest.fixture(autouse=True, scope="function")
def timer(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Pytest fixture that times the execution of each test.
    At the start of each test, it records the current time.
    After the test function completes, it calculates the run time and adds it to
    the test node's `user_properties`.

    Args:
        request: The `pytest.FixtureRequest` object through which the run time is stored
            in the test node's `user_properties`.
    """
    start_time = time.time()
    yield
    run_time = time.time() - start_time
    request.node.user_properties.append(("run_time", run_time))


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """
    Pytest hook that is called when a test report is being generated.
    It is used to generate and finalize reports for each test.

    Args:
        item: The test item for which the report is being generated.
        call: The call object from which the test result is retrieved.
    """
    challenge_data = item.funcargs.get("challenge_data", None)

    if not challenge_data:
        # this will only happen for dummy dependency setup tests
        return

    challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")

    if call.when == "call":
        answers = getattr(item, "answers", None)
        challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
        test_name = item.nodeid.split("::")[1]
        item.test_name = test_name

        generate_single_call_report(
            item, call, challenge_data, answers, challenge_location, test_name
        )

    if call.when == "teardown":
        finalize_reports(item, challenge_data)


def timeout_monitor(start_time: int) -> None:
    """
    Function that limits the total execution time of the test suite.
    This function is supposed to be run in a separate thread and calls `pytest.exit`
    if the total execution time has exceeded the global timeout.

    Args:
        start_time (int): The start time of the test suite.
    """
    while time.time() - start_time < GLOBAL_TIMEOUT:
        time.sleep(1)  # check every second

    pytest.exit("Test suite exceeded the global timeout", returncode=1)


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Pytest hook that is called at the start of a test session.

    Sets up and runs a `timeout_monitor` in a separate thread.
    """
    start_time = time.time()
    t = threading.Thread(target=timeout_monitor, args=(start_time,))
    t.daemon = True  # Daemon threads are abruptly stopped at shutdown
    t.start()


def pytest_sessionfinish(session: pytest.Session) -> None:
    """
    Pytest hook that is called at the end of a test session.

    Finalizes and saves the test reports.
    """
    session_finish(suite_reports)


@pytest.fixture
def scores(request: pytest.FixtureRequest) -> None:
    """
    Pytest fixture that retrieves the scores of the test class.
    The scores are retrieved from the `Challenge.scores` attribute
    using the test class name.

    Args:
        request: The request object.
    """
    test_class: type[Challenge] = request.node.cls
    return test_class.scores.get(test_class.__name__)


def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    """
    Pytest hook that is called after initial test collection has been performed.
    Modifies the collected test items based on the agent benchmark configuration,
    adding the dependency marker and category markers.

    Args:
        items: The collected test items to be modified.
        config: The active pytest configuration.
    """
    # agent_benchmark_config = load_agbenchmark_config()

    # regression_file = agent_benchmark_config.get_regression_reports_path()
    # data = (
    #     json.loads(open(regression_file, "r").read())
    #     if os.path.exists(regression_file)
    #     else {}
    # )

    for item in items:
        # Assuming item.cls is your test class
        test_class_instance: Challenge = item.cls()

        if "test_method" not in item.name:
            continue

        # Then you can access your properties
        name = item.parent.cls.__name__
        # dependencies = test_class_instance.data.dependencies

        # Filter dependencies if they are regression tests and this is an --improve sesh
        # if config.getoption("--improve") or config.getoption(
        #     "--category"
        # ):
        #     dependencies = [dep for dep in dependencies if not data.get(dep, None)]
        # if (
        #     config.getoption("--test")
        #     or config.getoption("--no_dep")
        #     or config.getoption("--maintain")
        # ):
        dependencies = test_class_instance.dependencies

        # Add depends marker dynamically
        item.add_marker(pytest.mark.depends(on=dependencies, name=name))

        categories = test_class_instance.data.category

        # Add category marker dynamically
        for category in categories:
            item.add_marker(category.value)
