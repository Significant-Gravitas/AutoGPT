import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--level", action="store", default=None, type=int, help="Specify test level"
    )


def pytest_configure(config: Config) -> None:
    config.option.level = config.getoption("--level")


@pytest.fixture
def user_selected_level(request: FixtureRequest) -> int:
    ## used for challenges in the goal oriented tests
    return request.config.option.level
