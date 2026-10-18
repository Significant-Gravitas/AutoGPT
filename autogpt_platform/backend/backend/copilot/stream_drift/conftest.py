"""The drift suite needs Redis and nothing else, so it opts out of SpinTestServer."""

import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():  # type: ignore[override]
    yield
