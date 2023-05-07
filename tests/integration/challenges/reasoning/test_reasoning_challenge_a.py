import contextlib
import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key
from typing import Generator, List


def create_events_file(
    temporal_reasoning_agent: Agent, file_name: str, events: List[str]
) -> None:
    """
    Writes a series of events to the file
    Args:
        temporal_reasoning_agent (Agent)
        file_name (str)
        events (List[str])
    """
    content = "\n".join(events)
    file_path = str(temporal_reasoning_agent.workspace.get_path(file_name))
    write_to_file(file_path, content)
    content = read_file(file_path)


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
def test_reasoning_challenge_a(temporal_reasoning_agent, monkeypatch) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs and checking the output file content.
    :param temporal_reasoning_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    input_sequence = ["s", "s", "s", "s", "s", "EXIT"]
    events = [
        "Cindy arrived at 8 PM",
        "Alice arrived 15 minutes before Cindy",
        "Eric was the last one to arrive",
        "David arrived 25 minutes later than Bob",
        "5 minutes after Alice arrived, Bob also arrived",
    ]
    create_events_file(temporal_reasoning_agent, "events.txt", events)
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(temporal_reasoning_agent, None)

    file_path = str(temporal_reasoning_agent.workspace.get_path("output.txt"))
    content = read_file(file_path)
    assert "Alice" in content, "Expected the file to contain Alice"
    assert "7:45" in content, "Expected the file to contain 7:45"
