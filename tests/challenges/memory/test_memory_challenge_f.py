import json
from typing import Dict

import pytest
from pytest_mock import MockerFixture

import yaml

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_interaction_loop

LEVEL_CURRENTLY_BEATEN = 1
MAX_LEVEL = 5
OUTPUT_LOCATION = "output.txt"

def read_yaml_file(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

@challenge()
def test_memory_challenge_f(
        memory_management_agent: Agent,
        patched_api_requestor: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        level_to_run: int,
        challenge_name: str,
) -> None:
    """
    The agent is given a series of events and must remember where are the marbles.
    Args:
        memory_management_agent (Agent)
        user_selected_level (int)
    """
    level_to_run=5
    # load the phrases from the YAML file
    sally_anne_test_phrases = read_yaml_file('./tests/challenges/memory/sally_anne_test_phrases.yaml')
    level_sally_anne_test_phrases = sally_anne_test_phrases[:level_to_run]
    create_instructions_files(
        memory_management_agent, level_to_run, level_sally_anne_test_phrases
    )
    run_interaction_loop(
        monkeypatch,
        memory_management_agent,
        level_to_run + 2,
        challenge_name,
        level_to_run,
    )

    file_path = get_workspace_path(memory_management_agent, OUTPUT_LOCATION)
    content = read_file(file_path, memory_management_agent)
    check_marble_location(content, level_to_run)


def check_marble_location(content: str, level: int) -> None:
    # Hold the true locations of the characters for each level
    expected_locations = {
        1: {"Anne": "inside", "Sally": "outside"},
        2: {"Anne": "inside", "Sally": "outside", "Bob": "outside", "Charlie": "outside"},
        3: {"Anne": "outside", "Sally": "outside", "Bob": "inside", "Charlie": "outside"},
        4: {"Anne": "inside", "Sally": "outside", "Bob": "inside", "Charlie": "outside"},
        5: {"Anne": "outside", "Sally": "outside", "Bob": "inside", "Charlie": "inside"},
        # Add the other expected locations here...
    }

    # Extract the beliefs from the AI's response
    ai_beliefs = extract_marbles_location(content)
    # Check the AI's beliefs against the expected beliefs
    for character, location in expected_locations[level].items():
        ai_character = ai_beliefs.get(character)
        assert (
                location == ai_character
        ), f"{character}, is in '{ai_character}'"


def extract_marbles_location(content: str) -> Dict[str, Dict[str, str]]:
    """Extract the beliefs of each character from the AI's output."""
    # Parse the JSON content
    content_dict = json.loads(content)
    return content_dict


def create_instructions_files(
        memory_management_agent: Agent,
        level: int,
        test_phrases: list,
        base_filename: str = "instructions_",
) -> None:
    """
    Creates a series of instructions files for the memory challenge.
    Args:
        level:
        memory_management_agent (Agent)
        test_phrases (list)
        base_filename (str, optional)
    """
    for i in range(1, level + 1):
        content = generate_content(i, test_phrases, base_filename, level)
        file_name = f"{base_filename}{i}.txt"
        file_path = get_workspace_path(memory_management_agent, file_name)
        write_to_file(file_path, content, memory_management_agent)


def generate_content(
        index: int, test_phrases: list, base_filename: str, level: int
) -> str:
    """
    Args:
        index: int
        test_phrases: list
        base_filename: str
        level: int
    Returns: str
    """
    phrase = test_phrases[index - 1]

    if index != level:
        return f"The current event is '{phrase}'.\nRead the file {base_filename}{index + 1}.txt using the read_file command."
    else:
        return f"""
        This event is '{phrase}'
        rules:
        1. Specify where is everyone at the end of the story (inside or outside the room)
        2. Do not create python script
        Instructions:
        I) Write the following information in the file output.txt in JSON format:
        The format should be as follows:
        {{
            "<character_name>": "<location>",
            ...
        }}
        Example(only use this to understand and learn how to use the format. Nothing else):
            Input:
                Anne is inside the room, charlie is outside the room 
            Output :
                {{"Anne": "inside", "Charlie": "inside"}}
        II) The file output.txt has not been created yet. You need to create it. After that, use the task_complete command.
        """
