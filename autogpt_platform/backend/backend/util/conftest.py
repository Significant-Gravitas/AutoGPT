"""Override session-scoped fixtures from parent conftest.py so unit tests
in this directory can run without the full server stack."""

import pytest


@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield
