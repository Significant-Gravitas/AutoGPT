import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path  # noqa
from typing import Any, Dict, Generator

import pytest

from agbenchmark.reports.reports import (
    finalize_reports,
    generate_single_call_report,
    session_finish,
)
from agbenchmark.utils.data_types import AgentBenchmarkConfig

GLOBAL_TIMEOUT = (
    1500  # The tests will stop after 25 minutes so we can send the reports.
)

pytest_plugins = ["agbenchmark.utils.dependencies"]
collect_ignore = ["challenges"]
suite_reports: dict[str, list] = {}


def load_config_from_request(request: Any) -> AgentBenchmarkConfig:
    """
    This function loads the configuration for the agent benchmark from a given request.

    Args:
        request (Any): The request object from which the agent benchmark configuration is to be loaded.

    Returns:
        AgentBenchmarkConfig: The loaded agent benchmark configuration.

    Raises:
        json.JSONDecodeError: If the benchmark configuration file is not a valid JSON file.
    """
    agent_benchmark_config_path = Path.cwd() / "agbenchmark_config" / "config.json"
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
            return agent_benchmark_config
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise


def resolve_workspace_path(workspace: Path) -> Path:
    """
    This function resolves the workspace path.

    Args:
        workspace (str): The workspace path which can be an absolute path or a path expression.

    Returns:
        str: The absolute path of the workspace.

    Raises:
        ValueError: If the workspace path expression is invalid.
    """
    if (
        isinstance(workspace, str)
        and workspace.startswith("${")
        and workspace.endswith("}")
    ):
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
    elif isinstance(workspace, str):
        return os.path.abspath(Path.cwd() / workspace)
    else:
        raise ValueError("Invalid workspace type. Expected str")


@pytest.fixture(scope="module")
def config(request: Any) -> Any:
    """
    This pytest fixture is responsible for loading the agent benchmark configuration from a given request.
    It also resolves the workspace path based on the configuration.
    This fixture is scoped to the module level, meaning it's invoked once per test module.

    Args:
        request (Any): The request object from which the agent benchmark configuration is to be loaded.

    Returns:
        Any: The loaded configuration dictionary.

    Raises:
        json.JSONDecodeError: If the benchmark configuration file is not a valid JSON file.
    """
    config = {"workspace": {}}
    agent_benchmark_config_path = Path.cwd() / "agbenchmark_config" / "config.json"
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise

    config["AgentBenchmarkConfig"] = agent_benchmark_config

    config["workspace"]["input"] = resolve_workspace_path(
        agent_benchmark_config.workspace.input
    )
    config["workspace"]["output"] = resolve_workspace_path(
        agent_benchmark_config.workspace.output
    )

    return config


@pytest.fixture(autouse=True)
def workspace(config: Dict[str, Any]) -> Generator[str, None, None]:
    """
    This pytest fixture is responsible for setting up and tearing down the workspace for each test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    The workspace path is retrieved from the configuration dictionary.
    If the workspace path does not exist, it is created.
    After the test function completes, the workspace is cleaned up unless 'keep_workspace_files' is set to True in the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary where the workspace path is defined.

    Yields:
        str: The workspace path.
    """
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
    if not config.get("keep_workspace_files", False):
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
    """
    This function is a pytest hook that is called to add command-line options.
    It is used to add custom command-line options that are specific to the agent benchmark tests.
    These options can be used to control the behavior of the tests.
    The "--mock" option is used to run the tests in mock mode.
    The "--api_mode" option is used to run the tests in API mode.
    The "--host" option is used to specify the host for the tests.
    The "--category" option is used to run only tests of a specific category.
    The "--nc" option is used to run the tests without caching.
    The "--cutoff" option is used to specify a cutoff time for the tests.
    The "--improve" option is used to run only the tests that are marked for improvement.
    The "--maintain" option is used to run only the tests that are marked for maintenance.
    The "--explore" option is used to run the tests in exploration mode.
    The "--test" option is used to run a specific test.
    The "--no_dep" option is used to run the tests without dependencies.

    Args:
        parser (Any): The parser object to which the command-line options are added.
    """
    parser.addoption("--no_dep", action="store_true", default=False)
    parser.addoption("--mock", action="store_true", default=False)
    parser.addoption("--api_mode", action="store_true", default=False)
    parser.addoption("--host", action="store_true", default=None)
    parser.addoption("--nc", action="store_true", default=False)
    parser.addoption("--cutoff", action="store_true", default=False)
    parser.addoption("--category", action="store_true", default=False)
    parser.addoption("--test", action="store_true", default=None)
    parser.addoption("--improve", action="store_true", default=False)
    parser.addoption("--maintain", action="store_true", default=False)
    parser.addoption("--explore", action="store_true", default=False)


@pytest.fixture(autouse=True)
def check_regression(request: Any) -> None:
    """
    This pytest fixture is responsible for checking if a test is a regression test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    The test name and the agent benchmark configuration are retrieved from the request object.
    The regression reports are loaded from the path specified in the agent benchmark configuration.
    If the "--improve" option is used and the test name exists in the regression tests, the test is skipped.
    If the "--maintain" option is used and the test name does not exist in the regression tests, the test is also skipped.

    Args:
        request (Any): The request object from which the test name and the agent benchmark configuration are retrieved.
    """
    test_name = request.node.parent.name
    agent_benchmark_config = load_config_from_request(request)
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
def challenge_data(request: Any) -> None:
    """
    This pytest fixture is responsible for providing the challenge data for each test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    The challenge data is retrieved from the request object's parameters.
    This fixture is essential for the pytest system as it provides the necessary data for each test.

    Args:
        request (Any): The request object from which the challenge data is retrieved.

    Returns:
        None: The challenge data is directly passed to the test function and does not need to be returned.
    """
    return request.param


@pytest.fixture(autouse=True, scope="session")
def mock(request: Any) -> None:
    """
    This pytest fixture is responsible for retrieving the value of the "--mock" command-line option.
    It is automatically used in every test session due to the 'autouse=True' parameter and 'session' scope.
    The "--mock" option is used to run the tests in mock mode.
    This fixture is essential for the pytest system as it provides the necessary command-line option value for each test session.

    Args:
        request (Any): The request object from which the "--mock" option value is retrieved.

    Returns:
        None: The "--mock" option value is directly passed to the test session and does not need to be returned.
    """
    return request.config.getoption("--mock")


@pytest.fixture(autouse=True, scope="function")
def timer(request: Any) -> Any:
    """
    This pytest fixture is responsible for timing the execution of each test.
    It is automatically used in every test due to the 'autouse=True' parameter and 'function' scope.
    At the start of each test, it records the current time.
    After the test function completes, it calculates the run time and appends it to the test node's user properties.
    This allows the run time of each test to be accessed later for reporting or analysis.

    Args:
        request (Any): The request object from which the test node is retrieved.

    Yields:
        None: Control is yielded back to the test function.
    """
    start_time = time.time()
    yield
    run_time = time.time() - start_time
    request.node.user_properties.append(("run_time", run_time))


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    """
    This function is a pytest hook that is called when a test report is being generated.
    It is used to generate and finalize reports for each test.

    Args:
        item (Any): The test item for which the report is being generated.
        call (Any): The call object from which the test result is retrieved.
    """
    challenge_data = item.funcargs.get("challenge_data", None)

    if not challenge_data:
        # this will only happen for dummy dependency setup tests
        return

    challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")

    flags = (
        "--test" in sys.argv
        or "--maintain" in sys.argv
        or "--improve" in sys.argv
        or "--explore" in sys.argv
    )

    if call.when == "call":
        generate_single_call_report(item, call, challenge_data)

    if call.when == "teardown":
        finalize_reports(item, challenge_data)


def timeout_monitor(start_time: int) -> None:
    """
    This function is responsible for monitoring the total execution time of the test suite.
    It runs in a separate thread and checks every second if the total execution time has exceeded the global timeout.
    If the global timeout is exceeded, it terminates the pytest session with a specific return code.

    Args:
        start_time (int): The start time of the test suite.
    """
    while time.time() - start_time < GLOBAL_TIMEOUT:
        time.sleep(1)  # check every second

    pytest.exit("Test suite exceeded the global timeout", returncode=1)


def pytest_sessionstart(session: Any) -> None:
    """
    This function is a pytest hook that is called at the start of the test session.
    It starts the timeout monitor in a separate thread.
    The timeout monitor checks if the total execution time of the test suite has exceeded the global timeout.

    Args:
        session (Any): The pytest session object.
    """
    start_time = time.time()
    t = threading.Thread(target=timeout_monitor, args=(start_time,))
    t.daemon = True  # Daemon threads are abruptly stopped at shutdown
    t.start()


def pytest_sessionfinish(session: Any) -> None:
    """
    This function is a pytest hook that is called at the end of the test session.
    It is used to finalize and save the test reports.
    The reports are saved in a specific location defined in the suite reports.

    Args:
        session (Any): The pytest session object.
    """
    session_finish(suite_reports)


@pytest.fixture
def scores(request: Any) -> None:
    """
    This pytest fixture is responsible for retrieving the scores of the test class.
    The scores are retrieved from the test class's 'scores' attribute using the test class name.
    This fixture is essential for the pytest system as it provides the necessary scores for each test.

    Args:
        request (Any): The request object from which the test class is retrieved.

    Returns:
        None: The scores are directly passed to the test function and do not need to be returned.
    """
    test_class_name = request.node.cls.__name__
    return request.node.cls.scores.get(test_class_name)


# this is adding the dependency marker and category markers automatically from the json
def pytest_collection_modifyitems(items: Any, config: Any) -> None:
    """
    This function is a pytest hook that is called after the test collection has been performed.
    It is used to modify the collected test items based on the agent benchmark configuration.
    The function loads the agent benchmark configuration from the specified path and retrieves the regression reports.
    For each test item, it checks if the test method exists and retrieves the dependencies and categories from the test class instance.
    If the "--improve" or "--category" options are used, the dependencies are filtered based on the regression data.
    If the "--test", "--no_dep", or "--maintain" options are used, the dependencies are cleared.
    The function then dynamically adds the 'depends' and 'category' markers to the test item.
    This function is essential for the pytest system as it provides the necessary modification of the test items based on the agent benchmark configuration.

    Args:
        items (Any): The collected test items to be modified.
        config (Any): The pytest configuration object from which the agent benchmark configuration path is retrieved.
    """
    agent_benchmark_config_path = str(Path.cwd() / "agbenchmark_config" / "config.json")
    try:
        with open(agent_benchmark_config_path) as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise

    regression_file = agent_benchmark_config.get_regression_reports_path()
    data = (
        json.loads(open(regression_file, "r").read())
        if os.path.exists(regression_file)
        else {}
    )

    for item in items:
        # Assuming item.cls is your test class
        test_class_instance = item.cls()

        if "test_method" not in item.name:
            continue

        # Then you can access your properties
        name = item.parent.cls.__name__
        # dependencies = test_class_instance.data.dependencies

        # Filter dependencies if they exist in regression data if its an improvement test
        # if config.getoption("--improve") or config.getoption(
        #     "--category"
        # ):
        #     dependencies = [dep for dep in dependencies if not data.get(dep, None)]
        # if (
        #     config.getoption("--test")
        #     or config.getoption("--no_dep")
        #     or config.getoption("--maintain")
        # ):
        dependencies = []

        # Add depends marker dynamically
        item.add_marker(pytest.mark.depends(on=dependencies, name=name))

        categories = test_class_instance.data.category

        # Add category marker dynamically
        for category in categories:
            item.add_marker(getattr(pytest.mark, category))


@pytest.fixture(scope="session", autouse=True)
def run_agent(request: Any) -> Any:
    """
    This pytest fixture is responsible for running the agent. It is automatically used in every test session due to the 'autouse=True' parameter and 'session' scope.
    If the "--api_mode" argument is not in the command line arguments, it starts a subprocess running the agbenchmark.
    The subprocess is terminated after the test session.
    If the "--api_mode" argument is present, it simply yields control back to the test session.
    This fixture is essential for the pytest system as it provides the necessary setup and teardown for running the agent in each test session.

    Args:
        request (Any): The request object from which the agent benchmark configuration path is retrieved.

    Yields:
        None: Control is yielded back to the test session.
    """
    if "--api_mode" not in sys.argv:
        command = [sys.executable, "-m", "agbenchmark.benchmarks"]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            # cwd=agent_benchmark_config_path.entry_path.parent.parent, # even if it's necessary to make it work, let's get rid ot that, this is too complex
        )
        time.sleep(3)
        yield
        print(f"Terminating agent")
        process.terminate()
    else:
        yield
