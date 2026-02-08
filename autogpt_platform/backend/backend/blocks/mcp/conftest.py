"""
Conftest for MCP block tests.

Override the session-scoped server and graph_cleanup fixtures from
backend/conftest.py so that MCP integration tests don't spin up the
full SpinTestServer infrastructure.
"""

import pytest


@pytest.fixture(scope="session")
def server():
    """No-op override — MCP tests don't need the full platform server."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup(server):
    """No-op override — MCP tests don't create graphs."""
    yield
