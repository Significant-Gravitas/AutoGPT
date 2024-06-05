import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Generator

import pytest

from agbenchmark.challenges import OPTIONAL_CATEGORIES, BaseChallenge
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types import Test
from agbenchmark.reports.ReportManager import RegressionTestsTracker
from agbenchmark.reports.reports import (
    add_test_result_to_report,
    make_empty_test_report,
    session_finish,
)
from agbenchmark.utils.data_types import Category

GLOBAL_TIMEOUT = (
    1500  # The tests will stop after 25 minutes so we can send the reports.
)

agbenchmark_config = AgentBenchmarkConfig.load()
logger = logging.getLogger(__name__)

pytest_plugins = ["agbenchmark.utils.dependencies"]
collect_ignore = ["challenges"]


@pytest.fixture(scope="module")
def config() -> AgentBenchmarkConfig:
    return agbenchmark_config


@pytest.fixture(autouse=True)
def temp_folder() -> Generator[Path, None, None]:
    """
    Pytest fixture that sets up and tears down the temporary folder for each test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    """

    # create output directory if it doesn't exist
    if not os.path.exists(agbenchmark_config.temp_folder):
        os.makedirs(agbenchmark_config.temp_folder, exist_ok=True)

    yield agbenchmark_config.temp_folder
    # teardown after test function completes
    if not os.getenv("KEEP_TEMP_FOLDER_FILES"):
        for filename in os.listdir(agbenchmark_config.temp_folder):
            file_path = os.path.join(agbenchmark_config.temp_folder, filename)
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
    * `--no-dep` is used to run the tests without dependencies.
    * `--keep-answers` is used to keep the answers of the tests.

    Args:
        parser: The Pytest CLI parser to which the command-line options are added.
    """
    parser.addoption("-N", "--attempts", action="store")
    parser.addoption("--no-dep", action="store_true")
    parser.addoption("--mock", action="store_true")
    parser.addoption("--host", default=None)
    parser.addoption("--nc", action="store_true")
    parser.addoption("--cutoff", action="store")
    parser.addoption("--category", action="append")
    parser.addoption("--test", action="append")
    parser.addoption("--improve", action="store_true")
    parser.addoption("--maintain", action="store_true")
    parser.addoption("--explore", action="store_true")
    parser.addoption("--keep-answers", action="store_true")


def pytest_configure(config: pytest.Config) -> None:
    # Register category markers to prevent "unknown marker" warnings
    for category in Category:
        config.addinivalue_line("markers", f"{category.value}: {category}")


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
    with contextlib.suppress(FileNotFoundError):
        rt_tracker = RegressionTestsTracker(agbenchmark_config.regression_tests_file)

        assert isinstance(request.node, pytest.Function)
        assert isinstance(request.node.parent, pytest.Class)
        test_name = request.node.parent.name
        challenge_location = getattr(request.node.cls, "CHALLENGE_LOCATION", "")
        skip_string = f"Skipping {test_name} at {challenge_location}"

        # Check if the test name exists in the regression tests
        is_regression_test = rt_tracker.has_regression_test(test_name)
        if request.config.getoption("--improve") and is_regression_test:
            pytest.skip(f"{skip_string} because it's a regression test")
        elif request.config.getoption("--maintain") and not is_regression_test:
            pytest.skip(f"{skip_string} because it's not a regression test")


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
    mock = request.config.getoption("--mock")
    assert isinstance(mock, bool)
    return mock


test_reports: dict[str, Test] = {}


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """
    Pytest hook that is called when a test report is being generated.
    It is used to generate and finalize reports for each test.

    Args:
        item: The test item for which the report is being generated.
        call: The call object from which the test result is retrieved.
    """
    challenge: type[BaseChallenge] = item.cls  # type: ignore
    challenge_id = challenge.info.eval_id

    if challenge_id not in test_reports:
        test_reports[challenge_id] = make_empty_test_report(challenge.info)

    if call.when == "setup":
        test_name = item.nodeid.split("::")[1]
        item.user_properties.append(("test_name", test_name))

    if call.when == "call":
        add_test_result_to_report(
            test_reports[challenge_id], item, call, agbenchmark_config
        )


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
    session_finish(agbenchmark_config)


def pytest_generate_tests(metafunc: pytest.Metafunc):
    n = metafunc.config.getoption("-N")
    metafunc.parametrize("i_attempt", range(int(n)) if type(n) is str else [0])


def pytest_collection_modifyitems(
    items: list[pytest.Function], config: pytest.Config
) -> None:
    """
    Pytest hook that is called after initial test collection has been performed.
    Modifies the collected test items based on the agent benchmark configuration,
    adding the dependency marker and category markers.

    Args:
        items: The collected test items to be modified.
        config: The active pytest configuration.
    """
    rt_tracker = RegressionTestsTracker(agbenchmark_config.regression_tests_file)

    try:
        challenges_beaten_in_the_past = json.loads(
            agbenchmark_config.challenges_already_beaten_file.read_bytes()
        )
    except FileNotFoundError:
        challenges_beaten_in_the_past = {}

    selected_tests: tuple[str] = config.getoption("--test")  # type: ignore
    selected_categories: tuple[str] = config.getoption("--category")  # type: ignore

    # Can't use a for-loop to remove items in-place
    i = 0
    while i < len(items):
        item = items[i]
        assert item.cls and issubclass(item.cls, BaseChallenge)
        challenge = item.cls
        challenge_name = challenge.info.name

        if not issubclass(challenge, BaseChallenge):
            item.warn(
                pytest.PytestCollectionWarning(
                    f"Non-challenge item collected: {challenge}"
                )
            )
            i += 1
            continue

        # --test: remove the test from the set if it's not specifically selected
        if selected_tests and challenge.info.name not in selected_tests:
            items.remove(item)
            continue

        # Filter challenges for --maintain, --improve, and --explore:
        # --maintain -> only challenges expected to be passed (= regression tests)
        # --improve -> only challenges that so far are not passed (reliably)
        # --explore -> only challenges that have never been passed
        is_regression_test = rt_tracker.has_regression_test(challenge.info.name)
        has_been_passed = challenges_beaten_in_the_past.get(challenge.info.name, False)
        if (
            (config.getoption("--maintain") and not is_regression_test)
            or (config.getoption("--improve") and is_regression_test)
            or (config.getoption("--explore") and has_been_passed)
        ):
            items.remove(item)
            continue

        dependencies = challenge.info.dependencies
        if (
            config.getoption("--test")
            or config.getoption("--no-dep")
            or config.getoption("--maintain")
        ):
            # Ignore dependencies:
            # --test -> user selected specific tests to run, don't care about deps
            # --no-dep -> ignore dependency relations regardless of test selection
            # --maintain -> all "regression" tests must pass, so run all of them
            dependencies = []
        elif config.getoption("--improve"):
            # Filter dependencies, keep only deps that are not "regression" tests
            dependencies = [
                d for d in dependencies if not rt_tracker.has_regression_test(d)
            ]

        # Set category markers
        challenge_categories = set(c.value for c in challenge.info.category)
        for category in challenge_categories:
            item.add_marker(category)

        # Enforce category selection
        if selected_categories:
            if not challenge_categories.intersection(set(selected_categories)):
                items.remove(item)
                continue
            # # Filter dependencies, keep only deps from selected categories
            # dependencies = [
            #     d for d in dependencies
            #     if not set(d.categories).intersection(set(selected_categories))
            # ]

        # Skip items in optional categories that are not selected for the subject agent
        challenge_optional_categories = challenge_categories & set(OPTIONAL_CATEGORIES)
        if challenge_optional_categories and not (
            agbenchmark_config.categories
            and challenge_optional_categories.issubset(
                set(agbenchmark_config.categories)
            )
        ):
            logger.debug(
                f"Skipping {challenge_name}: "
                f"category {' and '.join(challenge_optional_categories)} is optional, "
                "and not explicitly selected in the benchmark config."
            )
            items.remove(item)
            continue

        # Add marker for the DependencyManager
        item.add_marker(pytest.mark.depends(on=dependencies, name=challenge_name))

        i += 1
