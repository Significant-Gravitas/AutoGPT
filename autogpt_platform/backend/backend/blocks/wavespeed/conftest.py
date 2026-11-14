"""Keep the WaveSpeed unit tests independent of platform services.

Mirrors backend/blocks/dataforb2b/conftest.py: these tests only exercise the
block's request building and polling logic, so they have no reason to spin up
a test server or touch the database.
"""

from collections.abc import AsyncIterator

import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server() -> None:
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup() -> AsyncIterator[None]:
    yield
