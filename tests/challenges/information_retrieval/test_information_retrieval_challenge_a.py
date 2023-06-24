import pytest
from pytest_mock import MockerFixture

from autogpt.commands.file_operations import read_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path_from_agent, run_interaction_loop

CYCLE_COUNT = 3
EXPECTED_REVENUES = [["81"], ["81"], ["81", "53", "24", "21", "11", "7", "4", "3", "2"]]
from autogpt.agent import Agent

OUTPUT_LOCATION = "output.txt"


@challenge()
def test_information_retrieval_challenge_a(
    information_retrieval_agents: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    level_to_run: int,
    challenge_name: str,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    information_retrieval_agent = information_retrieval_agents[level_to_run - 1]
    run_interaction_loop(
        monkeypatch,
        information_retrieval_agent,
        CYCLE_COUNT,
        challenge_name,
        level_to_run,
    )

    file_path = get_workspace_path_from_agent(
        information_retrieval_agent, OUTPUT_LOCATION
    )
    content = read_file(file_path, information_retrieval_agent)
    expected_revenues = EXPECTED_REVENUES[level_to_run - 1]
    for revenue in expected_revenues:
        assert (
            f"{revenue}." in content or f"{revenue}," in content
        ), f"Expected the file to contain {revenue}"
