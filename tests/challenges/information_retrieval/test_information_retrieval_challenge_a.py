import pytest
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.basic_abilities.test_browse_website import USER_INPUTS
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_challenge

CYCLE_COUNT = 3
EXPECTED_REVENUES = [["81"], ["81"], ["81", "53", "24", "21", "11", "7", "4", "3", "2"]]

OUTPUT_LOCATION = "output.txt"
USER_INPUTS = [
    "Write to a file called output.txt containing tesla's revenue in 2022 after searching for 'tesla revenue 2022'.",
    "Write to a file called output.txt containing tesla's revenue in 2022.",
    "Write to a file called output.txt containing tesla's revenue every year since its creation.",
]


@challenge()
def test_information_retrieval_challenge_a(
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
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
    expected_revenues = EXPECTED_REVENUES[level_to_run - 1]
    for revenue in expected_revenues:
        assert (
            f"{revenue}." in content or f"{revenue}," in content
        ), f"Expected the file to contain {revenue}"
