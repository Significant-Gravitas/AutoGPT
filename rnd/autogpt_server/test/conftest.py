from multiprocessing import set_start_method

import pytest

from autogpt_server.util.test import SpinTestServer


@pytest.fixture(scope="session")
async def server():
    set_start_method("spawn", force=True)
    async with SpinTestServer() as server:
        yield server
