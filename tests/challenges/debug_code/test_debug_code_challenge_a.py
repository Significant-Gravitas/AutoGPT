from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from autogpt.agents import Agent
from autogpt.commands.execute_code import execute_python_file
from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import (
    copy_file_into_workspace,
    get_workspace_path,
    run_challenge,
)

CYCLE_COUNT = 5
EXPECTED_VALUES = ["[0, 1]", "[2, 5]", "[0, 3]"]
DIRECTORY_PATH = Path(__file__).parent / "data"
CODE_FILE_PATH = "code.py"
TEST_FILE_PATH = "test.py"
USER_INPUTS = [
    "1- Run test.py using the execute_python_file command.\n2- Read code.py using the read_file command.\n3- Modify code.py using the write_to_file command.\nRepeat step 1, 2 and 3 until test.py runs without errors. Do not modify the test.py file.",
    "1- Run test.py.\n2- Read code.py.\n3- Modify code.py.\nRepeat step 1, 2 and 3 until test.py runs without errors.\n",
    "Make test.py run without errors.",
]


@challenge()
def test_debug_code_challenge_a(
    dummy_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Test whether the agent can debug a simple code snippet.

    :param debug_code_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :patched_api_requestor: Sends api requests to our API CI pipeline
    :level_to_run: The level to run.
    """

    copy_file_into_workspace(workspace, DIRECTORY_PATH, CODE_FILE_PATH)
    copy_file_into_workspace(workspace, DIRECTORY_PATH, TEST_FILE_PATH)

    run_challenge(
        challenge_name,
        level_to_run,
        monkeypatch,
        USER_INPUTS[level_to_run - 1],
        CYCLE_COUNT,
    )

    output = execute_python_file(
        get_workspace_path(workspace, TEST_FILE_PATH),
        dummy_agent,
    )

    assert "error" not in output.lower(), f"Errors found in output: {output}!"

    for expected_value in EXPECTED_VALUES:
        assert (
            expected_value in output
        ), f"Expected output to contain {expected_value}, but it was not found in {output}!"
