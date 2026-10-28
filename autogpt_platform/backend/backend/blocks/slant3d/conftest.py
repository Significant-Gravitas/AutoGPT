import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server() -> None:
    return None
