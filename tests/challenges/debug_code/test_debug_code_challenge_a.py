from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.execute_code import execute_python_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import (
    copy_file_into_workspace,
    get_workspace_path_from_agent,
    run_interaction_loop,
)

CYCLE_COUNT = 5
EXPECTED_VALUES = ["[0, 1]", "[2, 5]", "[0, 3]"]
DIRECTORY_PATH = Path(__file__).parent / "data"
CODE_FILE_PATH = "code.py"
TEST_FILE_PATH = "test.py"


@challenge()
def test_debug_code_challenge_a(
    debug_code_agents: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
) -> None:
    """
    Test whether the agent can debug a simple code snippet.

    :param debug_code_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :patched_api_requestor: Sends api requests to our API CI pipeline
    :level_to_run: The level to run.
    """
    debug_code_agent = debug_code_agents[level_to_run - 1]

    copy_file_into_workspace(debug_code_agent, DIRECTORY_PATH, CODE_FILE_PATH)
    copy_file_into_workspace(debug_code_agent, DIRECTORY_PATH, TEST_FILE_PATH)

    run_interaction_loop(
        monkeypatch, debug_code_agent, CYCLE_COUNT, challenge_name, level_to_run
    )

    output = execute_python_file(
        get_workspace_path_from_agent(debug_code_agent, TEST_FILE_PATH),
        debug_code_agent,
    )

    assert "error" not in output.lower(), f"Errors found in output: {output}!"

    for expected_value in EXPECTED_VALUES:
        assert (
            expected_value in output
        ), f"Expected output to contain {expected_value}, but it was not found in {output}!"
