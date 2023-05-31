from typing import Any, Dict, Optional

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest

from tests.integration.challenges.challenge_decorator.challenge import Challenge
from tests.integration.conftest import BASE_VCR_CONFIG
from tests.vcr.vcr_filter import before_record_response


def before_record_response_filter_errors(
    response: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """In challenges we don't want to record errors (See issue #4461)"""
    if response["status"]["code"] >= 400:
        return None

    return before_record_response(response)


@pytest.fixture(scope="module")
def vcr_config() -> Dict[str, Any]:
    # this fixture is called by the pytest-recording vcr decorator.
    return BASE_VCR_CONFIG | {
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


@pytest.fixture(autouse=True)
def check_beat_challenges(request: FixtureRequest) -> None:
    Challenge.BEAT_CHALLENGES = request.config.getoption("--beat-challenges")
