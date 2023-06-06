import pytest
from pytest_mock import MockerFixture

from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3
EXPECTED_REVENUES = [["81"], ["81"], ["81", "53", "24", "21", "11", "7", "4", "3", "2"]]
from autogpt.agent import Agent


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_information_retrieval_challenge_a(
    information_retrieval_agents: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: MockerFixture,
    config: Config,
    level_to_run: int,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    information_retrieval_agent = information_retrieval_agents[level_to_run - 1]
    run_interaction_loop(monkeypatch, information_retrieval_agent, CYCLE_COUNT)

    file_path = str(information_retrieval_agent.workspace.get_path("output.txt"))
    content = read_file(file_path, config)
    expected_revenues = EXPECTED_REVENUES[level_to_run - 1]
    for revenue in expected_revenues:
        assert (
            f"{revenue}." in content or f"{revenue}," in content
        ), f"Expected the file to contain {revenue}"
