"""
Conftest for MCP block tests.

Override the session-scoped server and graph_cleanup fixtures from
backend/conftest.py so that MCP integration tests don't spin up the
full SpinTestServer infrastructure.
"""

import pytest
import pytest_asyncio


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring network")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip e2e tests unless --run-e2e is passed."""
    if not config.getoption("--run-e2e", default=False):
        skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e", action="store_true", default=False, help="run e2e tests"
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    """No-op override — MCP tests don't need the full platform server."""
    yield None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup(server):
    """No-op override — MCP tests don't create graphs."""
    yield
