import contextlib
from functools import wraps
import subprocess
from typing import Generator

import pytest

from autogpt.agent import agent
from autogpt.commands.execute_code import execute_python_file
from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.integration.challenges.utils import run_multiple_times
from tests.utils import requires_api_key
import docker

CODE = """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

# Example usage:
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)  # Output: [0, 1]
"""


def input_generator(input_sequence: list) -> Generator[str, None, None]:
    """
    Creates a generator that yields input strings from the given sequence.

    :param input_sequence: A list of input strings.
    :return: A generator that yields input strings.
    """
    yield from input_sequence


# @pytest.skip("Nobody beat this challenge yet")
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@run_multiple_times(3)
def test_debug_code_challenge_a(
    create_code_agent, monkeypatch
) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """

    # monkeypatch.setattr(docker, 'from_env', lambda: docker_client)
    client = subprocess.run(["docker.from_env()"], shell=True, capture_output=True)
    #client = docker.from_env()

    file_path = str(create_code_agent.workspace.get_path("code.py"))
    write_to_file(file_path, CODE)


    input_sequence = ["s", "s", "s", "s", "s", "EXIT"]
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        create_code_agent.start_interaction_loop()
    assert "[0, 1]" in execute_python_file(file_path), "Expected the output to be [0, 1]"

