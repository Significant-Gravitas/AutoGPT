import json
import os

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--level", action="store", default=None, type=int, help="Specify test level"
    )


def pytest_configure(config: Config) -> None:
    config.option.level = config.getoption("--level")


@pytest.fixture
def level_to_run(request: FixtureRequest) -> int:
    ## used for challenges in the goal oriented tests
    return request.config.option.level


import json
import os
from typing import Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup() -> Generator:
    if os.environ.get("CI") is None:
        yield
        return
    # setup
    cwd = os.getcwd()  # get current working directory
    new_score_filename = os.path.join(
        cwd, "tests/integration/challenges/new_score.json"
    )
    current_score_filename = os.path.join(
        cwd, "tests/integration/challenges/current_score.json"
    )

    # Remove the new_score.json at the start (in case it was left from previous run)
    if os.path.exists(new_score_filename):
        os.remove(new_score_filename)

    yield  # this will return the control to the test function

    # teardown
    if os.path.exists(new_score_filename):
        with open(new_score_filename, "r") as f_new:
            data = json.load(f_new)

        with open(current_score_filename, "w") as f_current:
            json.dump(data, f_current, indent=4)
        os.remove(new_score_filename)
