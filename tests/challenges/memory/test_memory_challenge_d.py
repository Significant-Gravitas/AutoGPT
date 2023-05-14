import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.integration.challenges.utils import get_level_to_run
from tests.utils import requires_api_key
import spacy
import json

LEVEL_CURRENTLY_BEATEN = 1
MAX_LEVEL = 5

@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_memory_challenge_d(
    memory_management_agent: Agent, user_selected_level: int
) -> None:
    """
    The agent is given a series of events and must remember the respective beliefs of the characters.
    Args:
        memory_management_agent (Agent)
        user_selected_level (int)
    """
    current_level = get_level_to_run(
        user_selected_level, LEVEL_CURRENTLY_BEATEN, MAX_LEVEL
    )
    sally_anne_test_phrases = [
        "Sally has a marble (marble A) and she puts it in her basket (basket S), then leaves the room. Anne moves marble A from Sally's basket (basket S) to her own basket (basket A).",
        "Sally gives a new marble (marble B) to Bob who is outside with her. Bob goes into the room and places marble B into Anne's basket (basket A). Anne tells Bob to tell Sally that he lost the marble b. Bob leaves the room and speaks to Sally about the marble B. Meanwhile, after Bob left the room, Anne moves marble A into the green box, but tells Charlie to tell Sally that marble A is under the sofa. Charlie leaves the room and speak to Sally about the marble A as instructed by Anne.",
        "Sally gives a new marble (marble C) to Charlie who is outside with her. Charlie enters the room and exchanges marble C with marble B in Anne's basket (basket A). Anne tells Charlie to tell Sally that he put marble C into the red box. Charlie leaves the room and speak to Sally about marble C as instructed by Anne. Meanwhile, after Charlie leaves the room, Bob enters into the room and moves marble A from the green box to under the sofa, but tells Anne to tell Sally that marble A is in the green box. Anne leaves the room and speak to Sally about the marble A as instructed by Bob",
        "Sally gives a new marble (marble D) to Anne. Anne gives the marble to Charlie. Charlie enters the room and gives marble D to Bob. Bob tells Charlie to tell Sally that he put marble D under the sofa. Bob put marble D under the sofa Charlie leaves the room and speaks to Sally about marble D. Meanwhile, after Charlie leaves the room, Bob takes marble A from under the sofa and places it in the blue box.",
        "Sally gives a new marble (marble E) to Charlie who is outside with her. Charlie enters the room and places marble E in the red box. Anne, who is already in the room, takes marble E from the red box, and hides it under the sofa. Then Anne leaves the room and tells Sally that marble E is in the green box. Meanwhile, after Anne leaves the room, Charlie who re-enters the room takes marble D from under the sofa and places it in his own basket (basket C)."
    ]

    level_sally_anne_test_phrases = sally_anne_test_phrases[:current_level]
    create_instructions_files(
        memory_management_agent, current_level, level_sally_anne_test_phrases
    )

    try:
        run_interaction_loop(memory_management_agent, 30*user_selected_level)
    except SystemExit:
        file_path = str(memory_management_agent.workspace.get_path("output.txt"))
        content = read_file(file_path)
        check_beliefs(content, current_level)

def check_beliefs(content, level):
    # Define the expected beliefs for each level
    expected_beliefs = {
        1: {
            'Sally': {
                'marble A': 'basket S',
            },
            'Anne': {
                'marble A': 'basket A',
            }
        },
        2: {
            'Sally': {
                'marble A': 'sofa',  # Because Charlie told her
            },
            'Anne': {
                'marble A': 'green box',  # Because she moved it there
                'marble B': 'basket A',  # Because Bob put it there and she was in the room
            },
            'Bob': {
                'B': 'basket A',  # Last place he put it
            },
            'Charlie': {
                'A': 'sofa',  # Because Anne told him to tell Sally so
            }
        },
        3: {
            'Sally': {
                'marble A': 'green box',  # Because Anne told her
                'marble C': 'red box',  # Because Charlie told her
            },
            'Anne': {
                'marble A': 'sofa',  # Because Bob moved it there and told her
                'marble B': 'basket A',  # Because Charlie exchanged marble C with marble B in her basket
                'marble C': 'basket A',  # Because Charlie exchanged marble C with marble B in her basket
            },
            'Bob': {
                'marble A': 'sofa',  # Because he moved it there
                'marble B': 'basket A',
                # Because Charlie exchanged marble C with marble B in Anne's basket, and he was in the room
                'marble C': 'basket A',
                # Because Charlie exchanged marble C with marble B in Anne's basket, and he was in the room
            },
            'Charlie': {
                'marble A': 'sofa',  # Last place he knew it was
                'marble B': 'basket A',  # Because he exchanged marble C with marble B in Anne's basket
                'marble C': 'red box',  # Because Anne told him to tell Sally so
            }
        },
        4: {
            'Sally': {
                'marble A': 'green box',  # Because Anne told her in the last conversation
                'marble C': 'red box',  # Because Charlie told her
                'marble D': 'sofa',  # Because Charlie told her
            },
            'Anne': {
                'marble A': 'blue box',  # Because Bob moved it there, and she was not in the room to see
                'marble B': 'basket A',  # Last place she knew it was
                'marble C': 'basket A',  # Last place she knew it was
                'marble D': 'sofa',  # Because Bob moved it there, and she was in the room to see
            },
            'Bob': {
                'marble A': 'blue box',  # Because he moved it there
                'marble B': 'basket A',  # Last place he knew it was
                'marble C': 'basket A',  # Last place he knew it was
                'marble D': 'sofa',  # Because he moved it there
            },
            'Charlie': {
                'marble A': 'sofa',  # Last place he knew it was
                'marble B': 'basket A',  # Last place he knew it was
                'marble C': 'red box',  # Last place he knew it was
                'marble D': 'sofa',  # Because Bob told him to tell Sally so
            }
        },
        5: {
        'Sally': {
            'marble A': 'green box',  # Because Anne told her in the last level
            'marble C': 'red box',  # Because Charlie told her
            'marble D': 'sofa',  # Because Charlie told her
            'marble E': 'green box',  # Because Anne told her
        },
        'Anne': {
            'marble A': 'blue box',  # Last place she knew it was
            'marble B': 'basket A',  # Last place she knew it was
            'marble C': 'basket A',  # Last place she knew it was
            'marble D': 'basket C',  # Last place she knew it was
            'marble E': 'sofa',  # Because she moved it there
        },
        'Charlie': {
            'marble A': 'blue box',  # Last place he knew it was
            'marble B': 'basket A',  # Last place he knew it was
            'marble C': 'basket A',  # Last place he knew it was
            'marble D': 'basket C',  # Because he moved it there
            'marble E': 'red box',  # Last place he knew it was
        },
        'Bob': {
            'marble A': 'blue box',  # Last place he knew it was
            'marble C': 'red box',  # Last place he knew it was
            'marble D': 'sofa',  # Last place he knew it was
        },
    },
    }
    # Extract the beliefs from the AI's response
    ai_beliefs = extract_beliefs(content)
    # Check the AI's beliefs against the expected beliefs
    for character, belief in expected_beliefs[level].items():
        assert ai_beliefs.get(character) == belief, f"For {character}, expected '{belief}' but got '{ai_beliefs.get(character)}'"

def extract_beliefs(content):
    """Extract the beliefs of each character from the AI's output."""
    # Parse the JSON content
    content_dict = json.loads(content)
    beliefs = content_dict.get('beliefs', {})
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
        file_path = str(memory_management_agent.workspace.get_path(file_name))
        write_to_file(file_path, content)


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
        5. Believed_location_of_the_specific_marble describes where is the marble like drawer, basket S, sofa, ...  Also, refer to the location by the name of the object (like 'sofa', 'drawer', etc.), not by its relative position (like 'under the sofa', 'in the drawer', etc.)
        6. Do not use expression like <Anne's basket> use <basket A> instead.
        7. Do not use expression like <under the sofa> use <sofa> instead.

        Instructions:
        
        I) Write the following information in the file output.txt in JSON format: 
        1. The respective beliefs of the characters (which means where every marble is according to character x, y, z. Character x should say where it believes every marble it is aware exist is)
  
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
