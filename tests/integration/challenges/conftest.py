import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--level", action="store", default=None, type=int, help="Specify test level"
    )


def pytest_configure(config):
    config.option.level = config.getoption("--level")


@pytest.fixture
def user_selected_level(request) -> int:
    ## used for challenges in the goal oriented tests
    return request.config.option.level
