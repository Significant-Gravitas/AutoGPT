import json
import os
import pytest
import shutil
from agbenchmark.tests.regression.RegressionManager import RegressionManager
import requests
from requests.exceptions import RequestException
from agbenchmark.mocks.MockManager import MockManager
from agbenchmark.challenges.define_task_types import ChallengeData


@pytest.fixture(scope="module")
def config():
    config_file = os.path.abspath("agbenchmark/config.json")
    print(f"Config file: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)
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


@pytest.fixture(autouse=True)
def server_response(request, config):
    """Calling to get a response"""
    if isinstance(request.param, tuple):
        task = request.param[0]  # The task is passed in indirectly
        mock_function_name = request.param[1]
    else:
        task = request.param
        mock_function_name = None
    # print(f"Server starting at {request.module}")
    # try:
    #     response = requests.post(
    #         f"{config['hostname']}:{config['port']}", data={"task": task}
    #     )
    #     response.raise_for_status()  # This will raise an HTTPError if the status is 4xx or 5xx
    # except RequestException:
    #     # If an exception occurs (could be connection, timeout, or HTTP errors), we use the mock

    if mock_function_name:
        mock_manager = MockManager(
            task
        )  # workspace doesn't need to be passed in, stays the same
        print("Server unavailable, using mock", mock_function_name)
        mock_manager.delegate(mock_function_name)
    else:
        print("No mock provided")

    # else:
    #     # This code is run if no exception occurred
    #     print(f"Request succeeded with status code {response.status_code}")


regression_json = "agbenchmark/tests/regression/regression_tests.json"

regression_manager = RegressionManager(regression_json)


# this is to get the challenge_data from every test
@pytest.fixture(autouse=True)
def regression_data(request):
    return request.param


def pytest_runtest_makereport(item, call):
    if call.when == "call":
        challenge_data = item.funcargs.get("regression_data", None)
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
    to add regression marker to collected test items."""
    for item in items:
        print("pytest_collection_modifyitems", item.nodeid)
        if item.nodeid.split("::")[1] in regression_manager.tests:
            print(regression_manager.tests)
            item.add_marker(pytest.mark.regression)


def pytest_sessionfinish():
    """Called at the end of the session to save regression tests"""
    regression_manager.save()
