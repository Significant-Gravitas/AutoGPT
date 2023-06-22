import json
import os
import pytest
import shutil
from agbenchmark.mocks.tests.retrieval_manual import mock_retrieval
from agbenchmark.tests.regression.RegressionManager import RegressionManager
import requests


@pytest.fixture(scope="module")
def config():
    config_file = os.path.abspath("agbenchmark/config.json")
    print(f"Config file: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


@pytest.fixture
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
    task = request.param  # The task is passed in indirectly
    print(f"Server starting at {request.module}")
    # response = requests.post(
    #     f"{config['hostname']}:{config['port']}", data={"task": task}
    # )
    # assert (
    #     response.status_code == 200
    # ), f"Request failed with status code {response.status_code}"
    mock_retrieval(task, config["workspace"])


regression_txt = "agbenchmark/tests/regression/regression_tests.txt"

regression_manager = RegressionManager(regression_txt)


def pytest_runtest_makereport(item, call):
    """Called for each test report. Generated for each stage
    of a test run (setup, call, teardown)."""
    if call.when == "call":
        if (
            call.excinfo is None
        ):  # if no error in the call stage, add it as a regression test
            regression_manager.add_test(item.nodeid)
        else:  # otherwise, :(
            regression_manager.remove_test(item.nodeid)


def pytest_collection_modifyitems(items):
    """Called once all test items are collected. Used
    to add regression marker to collected test items."""
    for item in items:
        print("pytest_collection_modifyitems", item.nodeid)
        if item.nodeid + "\n" in regression_manager.tests:
            print(regression_manager.tests)
            item.add_marker(pytest.mark.regression)


def pytest_sessionfinish():
    """Called at the end of the session to save regression tests"""
    regression_manager.save()
