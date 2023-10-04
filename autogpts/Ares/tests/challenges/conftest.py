from typing import Any, Dict, Generator, Optional

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge import Challenge
from tests.vcr import before_record_response


def before_record_response_filter_errors(
    response: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """In challenges we don't want to record errors (See issue #4461)"""
    if response["status"]["code"] >= 400:
        return None

    return before_record_response(response)


@pytest.fixture(scope="module")
def vcr_config(get_base_vcr_config: Dict[str, Any]) -> Dict[str, Any]:
    # this fixture is called by the pytest-recording vcr decorator.
    return get_base_vcr_config | {
        "before_record_response": before_record_response_filter_errors,
    }


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--level", action="store", default=None, type=int, help="Specify test level"
    )
    parser.addoption(
        "--beat-challenges",
        action="store_true",
        help="Spepcifies whether the test suite should attempt to beat challenges",
    )


def pytest_configure(config: Config) -> None:
    level = config.getoption("--level", default=None)
    config.option.level = level
    beat_challenges = config.getoption("--beat-challenges", default=False)
    config.option.beat_challenges = beat_challenges


@pytest.fixture
def level_to_run(request: FixtureRequest) -> int:
    ## used for challenges in the goal oriented tests
    return request.config.option.level


@pytest.fixture
def challenge_name() -> str:
    return Challenge.DEFAULT_CHALLENGE_NAME


@pytest.fixture(autouse=True)
def check_beat_challenges(request: FixtureRequest) -> None:
    Challenge.BEAT_CHALLENGES = request.config.getoption("--beat-challenges")


@pytest.fixture
def patched_make_workspace(mocker: MockerFixture, workspace: Workspace) -> Generator:
    def patched_make_workspace(*args: Any, **kwargs: Any) -> str:
        return workspace.root

    mocker.patch.object(
        Workspace,
        "make_workspace",
        new=patched_make_workspace,
    )

    yield
