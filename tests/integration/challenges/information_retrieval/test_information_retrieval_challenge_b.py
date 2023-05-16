import contextlib
from functools import wraps
from typing import Generator

import pytest

from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.integration.challenges.utils import run_multiple_times
from tests.utils import requires_api_key

###
# See https://github.com/Significant-Gravitas/Auto-GPT/issues/3836
# https://github.com/Significant-Gravitas/Auto-GPT/issues/3837 
#
# Case study to use: who was awarded the nobel prize in phsics in 2010?
# Answer: The Nobel Prize in Physics in 2010 was awarded to Andre Geim and Konstantin Novoselov for their groundbreaking experiments regarding the two-dimensional material graphene. Both scientists were at the University of Manchester, UK, when they made their discovery.
###


def input_generator(input_sequence: list) -> Generator[str, None, None]:
    """
    Creates a generator that yields input strings from the given sequence.

    :param input_sequence: A list of input strings.
    :return: A generator that yields input strings.
    """
    yield from input_sequence


# @pytest.skip("Nobody beat this challenge yet")
# @pytest.mark.skip("This challenge hasn't been beaten yet.")
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@run_multiple_times(3)
def test_information_retrieval_challenge_b(
    get_nobel_prize_agent, monkeypatch, patched_api_requestor
) -> None:
    """
    Test the challenge_b function in a given agent by mocking user inputs and checking the output file content.

    :param get_nobel_prize_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """

    # 's' is for running a self feedback command.
    input_sequence = ["y","y","y","y","y","y","EXIT"]
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(get_nobel_prize_agent, None)

    file_path = str(get_nobel_prize_agent.workspace.get_path("2010_nobel_prize_winners.txt"))
    content = read_file(file_path)
    assert "Andre Geim" in content, "Expected the file to contain Andre Geim"
    assert "Konstantin Novoselov" in content, "Expected the file to contain Konstantin Novoselov"
    assert "University of Manchester" in content, "Expected the file to contain University of Manchester"
    assert "graphene" in content, "Expected the file to contain graphene"


    #The Nobel Prize in Physics in 2010 was awarded to Andre Geim and Konstantin Novoselov for their groundbreaking experiments regarding the two-dimensional material graphene. Both scientists were at the University of Manchester, UK, when they made their discovery.
