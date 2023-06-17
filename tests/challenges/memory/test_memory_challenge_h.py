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
MAX_LEVEL = 4
OUTPUT_LOCATION = "output.txt"

def read_yaml_file(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

@challenge()
def test_memory_challenge_h(
    memory_management_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
) -> None:
    """
    The agent is given a series of events and must remember the respective beliefs of the characters.
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
    check_beliefs(content, level_to_run)


def check_beliefs(content: str, level: int) -> None:
    # Define the expected beliefs for each level
    expected_beliefs = {
        1: {
            "Sally": {
                "marbles": {
                    "marble A": ["basket S"],
                },
            },
            "Anne": {
                "marbles": {
                    "marble A": ["basket S", "basket A"],
                },
            },
        },
        2: {
            "Sally": {
                "marbles": {
                    "marble A": ["basket S"],
                    "marble B": [],
                },
            },
            "Anne": {
                "marbles": {
                    "marble A": ["basket S", "basket A", "green box"],
                    "marble B": ["basket A"],
                },
            },
            "Bob": {
                "marbles": {
                    "marble B": ["basket A"],
                },
            },
            "Charlie": {
                "marbles": {
                    "marble A": [],
                },
            },
        },
        3: {
            "Sally": {
                "marbles": {
                    "marble A": ["basket S"],
                    "marble C": [],
                },
            },
            "Anne": {
                "marbles": {
                    "marble A": ["basket S", "basket A", "green box"],
                    "marble B": ["basket A"],
                    "marble C": ["basket A"],
                },
            },
            "Bob": {
                "marbles": {
                    "marble A": ["green box", "sofa"],
                    "marble B": ["basket A"],
                },
            },
            "Charlie": {
                "marbles": {
                    "marble A": [],
                    "marble C": ["basket A"],
                },
            },
        },
        4: {
            "Sally": {
                "marbles": {
                    "marble A": ["basket S"],
                    "marble C": [],
                    "marble D": [],
                },
            },
            "Anne": {
                "marbles": {
                    "marble A": ["basket S", "basket A", "green box", "blue box"],
                    "marble B": ["basket A"],
                    "marble C": ["basket A"],
                    "marble D": [],
                },
            },
            "Bob": {
                "marbles": {
                    "marble A": ["green box", "sofa", "blue box"],
                    "marble B": ["basket A"],
                    "marble D": ["sofa"],
                },
            },
            "Charlie": {
                "marbles": {
                    "marble A": [],
                    "marble C": ["basket A"],
                    "marble D": ["sofa"],
                },
            },
        },
        # ...
        # Continue the structure for other levels
    }

    ai_beliefs = extract_beliefs(content)
    # Check the AI's beliefs against the expected beliefs
    for character, character_beliefs in expected_beliefs[level].items():
        for marble, expected_locations in character_beliefs["marbles"].items():
            ai_locations = ai_beliefs.get(character, {}).get("marbles", {}).get(marble, [])
            for location in expected_locations:
                assert (
                        location in ai_locations
                ), f"For {character}'s {marble}, expected '{location}' to be in '{ai_locations}'"

def extract_beliefs(content: str) -> Dict[str, Dict[str, str]]:
    """Extract the beliefs of each character from the AI's output."""
    # Parse the JSON content
    print (content)
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
        1. Everyone inside the room see and know what are the actions of the people inside the room.
        2. Everyone outside the room do not see and do not know the actions of the people inside the room.
        3. Only write about the marbles that are present in the level
        4. Marbles names are marble A, marble B, marble C, marble D, ...
        6. Do not use expression like <Anne's basket> use <basket A> instead.
        7. Do not create python script
        Instructions:
        I) Write the following information in the file output.txt in JSON format:
        1. The observation of every characters (which means what events have been seen by character x.)
        The format should be as follows:
        {{
                "<character_name 1>": {{
                    
                    "marbles":
                        "<marble_name 1>": ["<last location where the marble 1 has been seen by character 1>", "<new observed location where the marble 1 has been seen by character 1>", ...], 
                    ...          
                }},
                ...
        }}
        Example of output (only use this to understand and learn how to use the format. Nothing else):
        {{"Sally": {{"marbles": {{"marble A": ["sofa", "basket A", "basket S"], ...}}, "Anne": {{"marble B": ["basket S"], ...}}}}}}
        II) The file output.txt has not been created yet. You need to create it. After that, use the task_complete command.
        """
