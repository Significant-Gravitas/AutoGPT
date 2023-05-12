import contextlib
from functools import wraps
from typing import Generator

import pytest

from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.integration.challenges.utils import run_multiple_times
from tests.utils import requires_api_key


def input_generator(input_sequence: list) -> Generator[str, None, None]:
    """
    Creates a generator that yields input strings from the given sequence.

    :param input_sequence: A list of input strings.
    :return: A generator that yields input strings.
    """
    yield from input_sequence


# @pytest.skip("Nobody beat this challenge yet")
@pytest.mark.skip("This challenge hasn't been beaten yet.")
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@run_multiple_times(3)
def test_information_retrieval_challenge_a(
    get_company_revenue_agent, monkeypatch
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    input_sequence = ["s", "s", "s", "s", "s", "EXIT"]
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(get_company_revenue_agent, None)

    file_path = str(get_company_revenue_agent.workspace.get_path("output.txt"))
    content = read_file(file_path)
    assert "81" in content, "Expected the file to contain 81"
