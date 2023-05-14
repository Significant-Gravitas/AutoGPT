import contextlib
import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.utils import requires_api_key
from typing import Generator, List


def generate_instruction(
    index: int, base_event_filename: str, base_instruction_filename: str, num_files: int
) -> str:
    """
    Args:
        index: int
        base_filename: str
        num_files: int

    Returns: str
    """
    if index != num_files:
        return f"Read the event from {base_event_filename}{index}.txt and follow the instruction from {base_instruction_filename}{index + 1}.txt"
    return "Write the name and arrival time of the individual who reached first to output.txt file.\nShutdown"

def create_files(
    temporal_reasoning_agent: Agent,
    events: List[str],
    base_event_filename: str = "event_",
    base_instruction_filename: str = "instruction_"
) -> None:
    """
    Writes a series of event files and instruction files for the reasoning challenge
    Args:
        temporal_reasoning_agent (Agent)
        events (List[str])
        base_event_filename (str)
        base_instruction_filename (str)
    """
    num_files = len(events)
    for i in range(num_files + 1):
        instruction = generate_instruction(i, base_event_filename, base_instruction_filename, num_files)
        instruction_file_name = f"{base_instruction_filename}{i}.txt"
        instruction_file_path = str(temporal_reasoning_agent.workspace.get_path(instruction_file_name))
        write_to_file(instruction_file_path, instruction)
        if i < num_files:
            event_file_name = f"{base_event_filename}{i}.txt"
            event_file_path = str(temporal_reasoning_agent.workspace.get_path(event_file_name)) 
            write_to_file(event_file_path, events[i])


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
    input_sequence = ["y"] * 15 + ["EXIT"]
    events = [
        "Alice arrived 15 minutes before Cindy",
        "Eric was the last one to arrive",
        "David arrived 25 minutes later than Bob",
        "Cindy arrived at 8 PM",
        "5 minutes after Alice arrived, Bob also arrived",
    ]
    create_files(temporal_reasoning_agent, events)
    gen = input_generator(input_sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(temporal_reasoning_agent, None)

    file_path = str(temporal_reasoning_agent.workspace.get_path("output.txt"))
    content = read_file(file_path)
    assert "Alice" in content, "Expected the file to contain Alice"
    assert "7:45" in content, "Expected the file to contain 7:45"
