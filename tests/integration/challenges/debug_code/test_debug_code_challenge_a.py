from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.execute_code import execute_python_file
from autogpt.commands.file_operations import append_to_file, write_to_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_debug_code_challenge_a(
    debug_code_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    config: Config,
    level_to_run: int,
) -> None:
    """
    Test whether the agent can debug a simple code snippet.

    :param debug_code_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :patched_api_requestor: Sends api requests to our API CI pipeline
    :config: The config object for the agent.
    :level_to_run: The level to run.
    """

    file_path = str(debug_code_agent.workspace.get_path("code.py"))

    code_file_path = Path(__file__).parent / "data" / "two_sum.py"
    test_file_path = Path(__file__).parent / "data" / "two_sum_tests.py"

    write_to_file(file_path, code_file_path.read_text(), config)

    run_interaction_loop(monkeypatch, debug_code_agent, CYCLE_COUNT)

    append_to_file(file_path, test_file_path.read_text(), config)

    output = execute_python_file(file_path, config)
    assert "error" not in output.lower(), f"Errors found in output: {output}!"
