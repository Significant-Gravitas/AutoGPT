from typing import Any, AsyncGenerator
import pydevd_pycharm
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from autogpt.application import get_app

@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """
    Backend for anyio pytest plugin.

    :return: backend name.
    """
    return "asyncio"

@pytest.fixture
async def client(
    fastapi_app: FastAPI,
    anyio_backend: Any,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture that creates client for requesting server.

    :param fastapi_app: the application.
    :yield: client for the autogpt.
    """
    async with AsyncClient(app=fastapi_app, base_url="http://test") as ac:
        yield ac

def pytest_addoption(parser):
    parser.addoption(
        '--mode', action='store',
        default=None,
        help="""
        options : strict
        In this mode, if the response is different than the original one, the test fails.
        You use this mode when you refactor your code and you want to make sure you're not breaking anything
        """
    )
    parser.addoption("--docker-debug",         default=None,action="store_true", help="Enable PyCharm debugger")

def pytest_configure(config):
    if config.getoption("--docker-debug"):
        pydevd_pycharm.settrace('host.docker.internal', port=9731, stdoutToServer=True, stderrToServer=True, suspend=False)
#
@pytest.fixture
def fastapi_app() -> FastAPI:
    """
    Fixture for creating FastAPI autogpt.

    :return: fastapi autogpt with mocked dependencies.
    """
    return get_app()

@pytest.fixture()
def config(pytestconfig):
    return pytestconfig
