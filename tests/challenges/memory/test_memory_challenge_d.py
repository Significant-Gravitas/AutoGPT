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
def test_memory_challenge_d(
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
                "marble A": "basket S",
            },
            "Anne": {
                "marble A": "basket A",
            },
        },
        2: {
            "Sally": {
                "marble A": "sofa",  # Because Charlie told her
            },
            "Anne": {
                "marble A": "green box",  # Because she moved it there
                "marble B": "basket A",  # Because Bob put it there and she was in the room
            },
            "Bob": {
                "marble B": "basket A",  # Last place he put it
            },
            "Charlie": {
                "marble A": "sofa",  # Because Anne told him to tell Sally so
            },
        },
        3: {
            "Sally": {
                "marble A": "green box",  # Because Anne told her
                "marble C": "red box",  # Because Charlie told her
            },
            "Anne": {
                "marble A": "sofa",  # Because Bob moved it there and told her
                "marble B": "basket A",  # Because Charlie exchanged marble C with marble B in her basket
                "marble C": "basket A",  # Because Charlie exchanged marble C with marble B in her basket
            },
            "Bob": {
                "marble A": "sofa",  # Because he moved it there
                "marble B": "basket A",
                # Because Charlie exchanged marble C with marble B in Anne's basket, and he was in the room
                "marble C": "basket A",
                # Because Charlie exchanged marble C with marble B in Anne's basket, and he was in the room
            },
            "Charlie": {
                "marble A": "sofa",  # Last place he knew it was
                "marble B": "basket A",  # Because he exchanged marble C with marble B in Anne's basket
                "marble C": "red box",  # Because Anne told him to tell Sally so
            },
        },
        4: {
            "Sally": {
                "marble A": "green box",  # Because Anne told her in the last conversation
                "marble C": "red box",  # Because Charlie told her
                "marble D": "sofa",  # Because Charlie told her
            },
            "Anne": {
                "marble A": "blue box",  # Because Bob moved it there, and she was not in the room to see
                "marble B": "basket A",  # Last place she knew it was
                "marble C": "basket A",  # Last place she knew it was
                "marble D": "sofa",  # Because Bob moved it there, and she was in the room to see
            },
            "Bob": {
                "marble A": "blue box",  # Because he moved it there
                "marble B": "basket A",  # Last place he knew it was
                "marble C": "basket A",  # Last place he knew it was
                "marble D": "sofa",  # Because he moved it there
            },
            "Charlie": {
                "marble A": "sofa",  # Last place he knew it was
                "marble B": "basket A",  # Last place he knew it was
                "marble C": "red box",  # Last place he knew it was
                "marble D": "sofa",  # Because Bob told him to tell Sally so
            },
        },
        5: {
            "Sally": {
                "marble A": "green box",  # Because Anne told her in the last level
                "marble C": "red box",  # Because Charlie told her
                "marble D": "sofa",  # Because Charlie told her
                "marble E": "green box",  # Because Anne told her
            },
            "Anne": {
                "marble A": "blue box",  # Last place she knew it was
                "marble B": "basket A",  # Last place she knew it was
                "marble C": "basket A",  # Last place she knew it was
                "marble D": "basket C",  # Last place she knew it was
                "marble E": "sofa",  # Because she moved it there
            },
            "Charlie": {
                "marble A": "blue box",  # Last place he knew it was
                "marble B": "basket A",  # Last place he knew it was
                "marble C": "basket A",  # Last place he knew it was
                "marble D": "basket C",  # Because he moved it there
                "marble E": "red box",  # Last place he knew it was
            },
            "Bob": {
                "marble A": "blue box",  # Last place he knew it was
                "marble C": "red box",  # Last place he knew it was
                "marble D": "sofa",  # Last place he knew it was
            },
        },
    }

    # Extract the beliefs from the AI's response
    ai_beliefs = extract_beliefs(content)
    # Check the AI's beliefs against the expected beliefs
    for character, belief in expected_beliefs[level].items():
        for marble, location in belief.items():
            ai_belief = ai_beliefs.get(character, {}).get(marble, "")
            assert (
                location in ai_belief
            ), f"For {character}'s {marble}, expected '{location}' to be in '{ai_belief}'"


def extract_beliefs(content: str) -> Dict[str, Dict[str, str]]:
    """Extract the beliefs of each character from the AI's output."""
    # Parse the JSON content
    content_dict = json.loads(content)
    beliefs = content_dict.get("beliefs", {})
    return beliefs


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
        5. Believed_location_of_the_specific_marble describes where is the marble like drawer, basket S, sofa, ...
        6. Do not use expression like <Anne's basket> use <basket A> instead.
        7. All characters believe what they're being told by other characters.
        8. Do not mention the marble if the character does not know where it is.
        9. Do not create python script
        Instructions:
        I) Write the following information in the file output.txt in JSON format:
        1. The respective beliefs of the characters (which means where every marble is according to character x, y, z.)
        The format should be as follows:
        {{
            "beliefs": {{
                "<character_name>": {{
                    "<marble_name>": "<believed_location_of_the_specific_marble>",
                    ...
                }},
                ...
            }},
        }}
        Example of output (only use this to understand and learn how to use the format. Nothing else):
        {{"beliefs": {{"Sally": {{"marble A": "basket A"}}, "Bob": {{"marble B": "basket S"}}, "Anne": {{"marble A": "green box"}}, "Charlie": {{"marble B": "sofa"}}}}
        II) The file output.txt has not been created yet. You need to create it. After that, use the task_complete command.
        """
