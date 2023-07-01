import pytest
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_challenge

CYCLE_COUNT = 3
OUTPUT_LOCATION = "2010_nobel_prize_winners.txt"
USER_INPUTS = [
    "Write to file the winner's name(s), affiliated university, and discovery of the 2010 nobel prize in physics. Write your final answer to 2010_nobel_prize_winners.txt."
]


@challenge()
def test_information_retrieval_challenge_b(
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Test the challenge_b function in a given agent by mocking user inputs and checking the output file content.

    :param get_nobel_prize_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :param patched_api_requestor: APIRequestor Patch to override the openai.api_requestor module for testing.
    :param level_to_run: The level to run.
    """

    run_challenge(
        challenge_name,
        level_to_run,
        monkeypatch,
        USER_INPUTS[level_to_run - 1],
        CYCLE_COUNT,
    )

    file_path = get_workspace_path(workspace, OUTPUT_LOCATION)

    with open(file_path, "r") as file:
        content = file.read()
    assert "Andre Geim" in content, "Expected the file to contain Andre Geim"
    assert (
        "Konstantin Novoselov" in content
    ), "Expected the file to contain Konstantin Novoselov"
    assert (
        "University of Manchester" in content
    ), "Expected the file to contain University of Manchester"
    assert "graphene" in content, "Expected the file to contain graphene"
