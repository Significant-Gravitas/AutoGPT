import contextlib

import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_interaction_loop

CYCLE_COUNT = 3
OUTPUT_LOCATION = "2010_nobel_prize_winners.txt"


@challenge()
def test_information_retrieval_challenge_b(
    get_nobel_prize_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
) -> None:
    """
    Test the challenge_b function in a given agent by mocking user inputs and checking the output file content.

    :param get_nobel_prize_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :param patched_api_requestor: APIRequestor Patch to override the openai.api_requestor module for testing.
    :param level_to_run: The level to run.
    """

    with contextlib.suppress(SystemExit):
        run_interaction_loop(monkeypatch, get_nobel_prize_agent, CYCLE_COUNT)
    file_path = get_workspace_path(get_nobel_prize_agent, OUTPUT_LOCATION)

    content = read_file(file_path, get_nobel_prize_agent)
    assert "Andre Geim" in content, "Expected the file to contain Andre Geim"
    assert (
        "Konstantin Novoselov" in content
    ), "Expected the file to contain Konstantin Novoselov"
    assert (
        "University of Manchester" in content
    ), "Expected the file to contain University of Manchester"
    assert "graphene" in content, "Expected the file to contain graphene"
