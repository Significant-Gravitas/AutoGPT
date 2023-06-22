from typing import AsyncGenerator, Any

import pytest

from autogpt.api.application import get_app
from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, setup_mock_input, setup_mock_log_cycle_agent_name
from fastapi import FastAPI
from httpx import AsyncClient

CYCLE_COUNT_PER_LEVEL = [1, 1]
EXPECTED_OUTPUTS_PER_LEVEL = [
    {"hello_world.txt": ["Hello World"]},
    {"hello_world_1.txt": ["Hello World"], "hello_world_2.txt": ["Hello World"]},
]
USER_REQUESTS = [
    "Write 'Hello World' into a file named \"hello_world.txt\".",
    'Write \'Hello World\' into 2 files named "hello_world_1.txt"and "hello_world_2.txt".',
]

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
    :yield: client for the app.
    """
    async with AsyncClient(app=fastapi_app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def fastapi_app() -> FastAPI:
    """
    Fixture for creating FastAPI app.

    :return: fastapi app with mocked dependencies.
    """
    return get_app()

@challenge()
async def test_write_file(
    workspace: Workspace,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    client: AsyncClient,
) -> None:
    user_request = USER_REQUESTS[level_to_run - 1]

    import json

    import requests

    # Define the URL
    url = "http://0.0.0.0:6060/api/v1/agents"

    # Define the headers for the POST request
    headers = {"Content-Type": "application/json"}
    # "content": user_request,
    # Define the data payload for the POST request
    data = {
        "workspace": {
            "configuration": {
                "root": str(workspace.root)
            }
        }
    }
    setup_mock_input(monkeypatch, CYCLE_COUNT_PER_LEVEL[level_to_run - 1])
    setup_mock_log_cycle_agent_name(monkeypatch, challenge_name, level_to_run)
    # Make the POST request and store the response
    response = await client.post(url, json = data, headers=headers)

    json_response = response.json()

    url = f"http://0.0.0.0:6060/api/v1/agents/{json_response['agent_id']}/interactions"
    data = {
        "workspace": { # TODO remove this once we have a small sqlite db that saves the workspace location
            "configuration": {
                "root": str(workspace.root)
            }
        },
        "user_input": user_request
    }

    await client.post(url, json = data, headers=headers)

    expected_outputs = EXPECTED_OUTPUTS_PER_LEVEL[level_to_run - 1]

    for file_name, expected_lines in expected_outputs.items():
        file_path = get_workspace_path(workspace, file_name)
        with open(file_path, 'r') as file:
            # Read the file
            content = file.read()

        for expected_line in expected_lines:
            assert (
                expected_line in content
            ), f"Expected '{expected_line}' in file {file_name}, but it was not found"
