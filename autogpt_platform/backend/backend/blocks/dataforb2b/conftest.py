"""Keep DataForB2B unit tests independent of platform services."""

from collections.abc import AsyncIterator

import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server() -> None:
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup() -> AsyncIterator[None]:
    yield
