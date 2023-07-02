import pytest
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_challenge

CYCLE_COUNT = 3
COO = [["Luke Lafreniere"], ["Luke Lafreniere"], ["Luke Lafreniere 2017"]]

OUTPUT_LOCATION = "output.txt"
USER_INPUTS = [
    "Write to a file called output.txt containing the name and title of the current Chief Operating Officer of Floatplane Media.",
    "Write to a file called output.txt containing the name and title of the current Chief Operating Officer of https://www.floatplane.com.",
    "Write to a file called output.txt containing the name and title of the current Chief Operating Officer of https://www.floatplane.com and the year it was formed.",
]


@challenge()
def test_information_retrieval_challenge_c(
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Test the challenge_c function in a given agent by mocking user inputs and checking the output file content.

    :param get_floatplane_ceo_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
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
    coo_name = COO[level_to_run - 1]
    for chief in coo_name:
        assert chief in content, f"Expected the file to contain {chief}"
