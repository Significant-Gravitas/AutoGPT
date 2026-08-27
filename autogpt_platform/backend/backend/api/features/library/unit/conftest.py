import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield
