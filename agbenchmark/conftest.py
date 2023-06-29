import json
import os
import pytest
import shutil
import subprocess
import sys
from agbenchmark.tests.regression.RegressionManager import RegressionManager
from agbenchmark.mocks.MockManager import MockManager
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="module")
def config(request):
    config_file = os.path.abspath("agbenchmark/config.json")
    print(f"Config file: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    if request.config.getoption("--mock"):
        config["workspace"] = "agbenchmark/mocks/workspace"

    return config


@pytest.fixture(scope="module")
def workspace(config):
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


def pytest_addoption(parser):
    parser.addoption("--mock", action="store_true", default=False)


def check_cycle_count(cycle_count: int, cutoff: int, proc):
    """Increment, print, and check cycle count."""
    cycle_count += 1
    print(f"Cycle count: {cycle_count}")
    if cycle_count >= cutoff:
        proc.terminate(force=True)
    return cycle_count


AGENT_NAME = os.getenv("AGENT_NAME")


@pytest.fixture(autouse=True)
def run_agent(request, config):
    """Calling to get a response"""
    if isinstance(request.param, tuple):
        task = request.param[0]  # The task is passed in indirectly
        mock_function_name = request.param[1] or None
    else:
        task = request.param
        mock_function_name = None

    if mock_function_name != None and (request.config.getoption("--mock")):
        if mock_function_name:
            mock_manager = MockManager(
                task
            )  # workspace doesn't need to be passed in, stays the same
            print("Server unavailable, using mock", mock_function_name)
            mock_manager.delegate(mock_function_name)
        else:
            print("No mock provided")
    else:
        path = os.path.join(os.getcwd(), f"agent\\{AGENT_NAME}")

        timeout = sys.maxsize

        if config["cutoff"]["type"] == "time":
            timeout = config["cutoff"]["count"] or 60
            
        from pexpect.popen_spawn import PopenSpawn

        print(f"Running {task} with timeout {timeout}")

        # Starting the subprocess using pexpect
        proc = PopenSpawn("python", ["miniagi.py", task], timeout=timeout, cwd=path)

        print("proc", proc)

        cycle_count = 0

        while True:
            try:
                # If we get the prompt for user input, we send "\n"
                if config["cutoff"]["type"] == "user_input":
                    proc.expect([config["cutoff"]["user_prompt"]])
                    proc.sendline(config["cutoff"]["user_input"])
                    cycle_count = check_cycle_count(
                        cycle_count, config["cutoff"]["count"], proc
                    )
                elif config["cutoff"]["type"] == "cycle_count":
                    match = proc.expect([r"Cycle count: (\d+)"])
                    if match is not None:
                        cycle_count = int(match.group(1))  # type: ignore
                        cycle_count = check_cycle_count(
                            cycle_count, config["cutoff"]["count"], proc
                        )

                # for cutoff type "time", just let it run until timeout
            except expect.TIMEOUT:
                print("The subprocess has exceeded the time limit and was terminated.")
                break
            except expect.EOF:
                print("The subprocess has finished running.")
                break

        proc.close()


regression_json = "agbenchmark/tests/regression/regression_tests.json"

regression_manager = RegressionManager(regression_json)


# this is to get the challenge_data from every test
@pytest.fixture(autouse=True)
def challenge_data(request):
    return request.param


def pytest_runtest_makereport(item, call):
    if call.when == "call":
        challenge_data = item.funcargs.get("challenge_data", None)
        difficulty = challenge_data.info.difficulty if challenge_data else "unknown"
        dependencies = challenge_data.dependencies if challenge_data else []

        test_details = {
            "difficulty": difficulty,
            "dependencies": dependencies,
            "test": item.nodeid,
        }

        print("pytest_runtest_makereport", test_details)
        if call.excinfo is None:
            regression_manager.add_test(item.nodeid.split("::")[1], test_details)
        else:
            regression_manager.remove_test(item.nodeid.split("::")[1])


def pytest_collection_modifyitems(items):
    """Called once all test items are collected. Used
    to add regression and depends markers to collected test items."""
    for item in items:
        # regression add
        if item.nodeid.split("::")[1] in regression_manager.tests:
            print(regression_manager.tests)
            item.add_marker(pytest.mark.regression)


def pytest_sessionfinish():
    """Called at the end of the session to save regression tests"""
    regression_manager.save()


# this is so that all tests can inherit from the Challenge class
def pytest_generate_tests(metafunc):
    if "challenge_data" in metafunc.fixturenames:
        # Get the instance of the test class
        test_class = metafunc.cls()

        # Generate the parameters
        params = test_class.data

        # Add the parameters to the test function
        metafunc.parametrize("challenge_data", [params], indirect=True)

    if "run_agent" in metafunc.fixturenames:
        # Get the instance of the test class
        test_class = metafunc.cls()

        # Generate the parameters
        params = [(test_class.task, test_class.mock)]

        # Add the parameters to the test function
        metafunc.parametrize("run_agent", params, indirect=True)
