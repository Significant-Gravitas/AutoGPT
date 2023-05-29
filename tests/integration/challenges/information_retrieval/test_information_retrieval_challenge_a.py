import pytest

from autogpt.commands.file_operations import read_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key

CYCLE_COUNT = 3
from autogpt.agent import Agent


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_information_retrieval_challenge_a(
    get_company_revenue_agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    patched_api_requestor: None,
    config: Config,
    level_to_run: int,
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    run_interaction_loop(monkeypatch, get_company_revenue_agent, CYCLE_COUNT)

    file_path = str(get_company_revenue_agent.workspace.get_path("output.txt"))
    content = read_file(file_path, config)
    assert "81" in content, "Expected the file to contain 81"
