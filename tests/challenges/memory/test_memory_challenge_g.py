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
def test_memory_challenge_g(
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
    check_marble_history(content, level_to_run)


def check_marble_history(content: str, level: int) -> None:
    # Hold the true locations of the characters for each level
    expected_locations = {
    1: {
        "marble A": ["basket S", "basket A"]
    },
    2: {
        "marble A": ["basket S", "basket A", "green box"],
        "marble B": ["basket A"]
    },
    3: {
        "marble A": ["basket S", "basket A", "green box", "sofa"],
        "marble B": ["basket A"],
        "marble C": ["basket A"]
    },
    4: {
        "marble A": ["basket S", "basket A", "green box", "sofa", "blue box"],
        "marble B": ["basket A"],
        "marble C": ["basket A"],
        "marble D": ["sofa"]
    },
    5: {
        "marble A": ["basket S", "basket A", "green box", "sofa", "blue box"],
        "marble B": ["basket A"],
        "marble C": ["basket A"],
        "marble D": ["sofa", "basket C"],
        "marble E": ["red box", "sofa"]
    }
    }

    # Extract the beliefs from the AI's response
    ai_beliefs = extract_marbles_location(content)

    # Check the AI's beliefs against the expected beliefs
    for marble, locations in expected_locations[level].items():
        ai_marble_locations = ai_beliefs.get(marble, [])

        assert (
            all(location in ai_marble_locations for location in locations)
        ), f"For {marble}, expected locations were {locations} but got {ai_marble_locations}"


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
        1. Describe the history location of each marble in chronological order.
        2. Do not create python script
        Instructions:
        I) Write the following information in the file output.txt in JSON format:
        The format should be as follows:
        {{
            "<marble_name>": ["<location1>", "<location2>", "<location3>",
            ...]
        }}
        Example(only use this to understand and learn how to use the format. Nothing else):
            Input:
                Anne is inside the room, charlie is outside the room 
            Output :
                {{"marble A": "red box", "sofa", "basket A"], 
                "marble B": ...
                }}
        II) The file output.txt has not been created yet. You need to create it. After that, use the task_complete command.
        """
