"""Keep Local PC route unit tests independent of platform services."""

import pytest


@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield
