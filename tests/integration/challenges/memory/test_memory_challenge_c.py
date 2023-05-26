import pytest

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from autogpt.config import Config
from tests.integration.challenges.utils import (
    generate_noise,
    get_level_to_run,
    run_interaction_loop,
)
from tests.utils import requires_api_key

LEVEL_CURRENTLY_BEATEN = -1
MAX_LEVEL = 5
NOISE = 1000


# @pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_memory_challenge_c(
    memory_management_agent: Agent,
    user_selected_level: int,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
) -> None:
    """
    Instead of reading task Ids from files as with the previous challenges, the agent now must remember
    phrases which may have semantically similar meaning and the agent must write the phrases to a file
    after seeing several of them.

    Args:
        memory_management_agent (Agent)
        user_selected_level (int)
    """
    current_level = get_level_to_run(
        user_selected_level, LEVEL_CURRENTLY_BEATEN, MAX_LEVEL
    )
    silly_phrases = [
        "The purple elephant danced on a rainbow while eating a taco.",
        "The sneaky toaster stole my socks and ran away to Hawaii.",
        "My pet rock sings better than Beyoncé on Tuesdays.",
        "The giant hamster rode a unicycle through the crowded mall.",
        "The talking tree gave me a high-five and then flew away.",
        "I have a collection of invisible hats that I wear on special occasions.",
        "The flying spaghetti monster stole my sandwich and left a note saying 'thanks for the snack!'",
        "My imaginary friend is a dragon who loves to play video games.",
        "I once saw a cloud shaped like a giant chicken eating a pizza.",
        "The ninja unicorn disguised itself as a potted plant and infiltrated the office.",
    ]

    level_silly_phrases = silly_phrases[:current_level]
    create_instructions_files(
        memory_management_agent, current_level, level_silly_phrases, config=config
    )

    run_interaction_loop(monkeypatch, memory_management_agent, current_level + 2)

    file_path = str(memory_management_agent.workspace.get_path("output.txt"))
    content = read_file(file_path)
    for phrase in level_silly_phrases:
        assert phrase in content, f"Expected the file to contain {phrase}"


def create_instructions_files(
    memory_management_agent: Agent,
    level: int,
    task_ids: list,
    config: Config,
    base_filename: str = "instructions_",
) -> None:
    """
    Creates a series of instructions files for the memory challenge.
    Args:
        level:
        memory_management_agent (Agent)
        num_files (int)
        task_ids (list)
        base_filename (str, optional)
    """
    for i in range(1, level + 1):
        content = generate_content(i, task_ids, base_filename, level)
        file_name = f"{base_filename}{i}.txt"
        file_path = str(memory_management_agent.workspace.get_path(file_name))
        write_to_file(file_path, content, config)


def generate_content(
    index: int, silly_phrases: list, base_filename: str, level: int
) -> str:
    """
    Args:
        index: int
        task_ids: list
        base_filename: str
        num_files: int

    Returns: str
    """
    phrase = silly_phrases[index - 1]
    noise = generate_noise(NOISE)
    if index != level:
        if level == 1:
            return f"{noise}\nThe current phrase to remember is '{phrase}'.\n{noise}\nWrite all the phrases into the file output.txt. The file has not been created yet. After that, use the task_complete command."
        return f"{noise}\nThe current phrase is '{phrase}'.\n{noise}\nRead the file {base_filename}{index + 1}.txt using the read_file command."
    return f"{noise}\nThis phrase is '{phrase}'\n{noise}\nWrite all the phrases into the file output.txt. The file has not been created yet. After that, use the task_complete command.\n"
