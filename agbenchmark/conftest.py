import json
import os
import pytest
import shutil
from agbenchmark.tests.regression.RegressionManager import RegressionManager


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
