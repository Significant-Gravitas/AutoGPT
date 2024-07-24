import pytest

from autogpt_server.util.test import SpinTestServer


@pytest.fixture(scope="session")
async def server():
    async with SpinTestServer() as server:
        yield server
