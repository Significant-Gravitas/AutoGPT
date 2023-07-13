import pytest
from pytest_mock import MockerFixture

from autogpt.commands.file_operations import read_file
from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import generate_noise, get_workspace_path, run_challenge

NOISE = 1200
OUTPUT_LOCATION = "output.txt"
USER_INPUT = "Use the command read_file to read the instructions_1.txt file\nFollow the instructions in the instructions_1.txt file"


@challenge()
def test_memory_challenge_c(
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    Instead of reading task Ids from files as with the previous challenges, the agent now must remember
    phrases which may have semantically similar meaning and the agent must write the phrases to a file
    after seeing several of them.

    Args:
        workspace (Workspace)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    silly_phrases = [
        "The purple elephant danced on a rainbow while eating a taco",
        "The sneaky toaster stole my socks and ran away to Hawaii",
        "My pet rock sings better than Beyoncé on Tuesdays",
        "The giant hamster rode a unicycle through the crowded mall",
        "The talking tree gave me a high-five and then flew away",
        "I have a collection of invisible hats that I wear on special occasions",
        "The flying spaghetti monster stole my sandwich and left a note saying 'thanks for the snack'",
        "My imaginary friend is a dragon who loves to play video games",
        "I once saw a cloud shaped like a giant chicken eating a pizza",
        "The ninja unicorn disguised itself as a potted plant and infiltrated the office",
    ]

    level_silly_phrases = silly_phrases[:level_to_run]
    create_instructions_files(
        workspace,
        level_to_run,
        level_silly_phrases,
    )

    run_challenge(
        challenge_name, level_to_run, monkeypatch, USER_INPUT, level_to_run + 2
    )

    file_path = get_workspace_path(workspace, OUTPUT_LOCATION)
    content = read_file(file_path, agent=workspace)
    for phrase in level_silly_phrases:
        assert phrase in content, f"Expected the file to contain {phrase}"


def create_instructions_files(
    workspace: Workspace,
    level: int,
    task_ids: list,
    base_filename: str = "instructions_",
) -> None:
    """
    Creates a series of instructions files for the memory challenge.
    Args:
        level:
        workspace (Workspace)
        num_files (int)
        task_ids (list)
        base_filename (str, optional)
    """
    for i in range(1, level + 1):
        content = generate_content(i, task_ids, base_filename, level)
        file_name = f"{base_filename}{i}.txt"
        file_path = get_workspace_path(workspace, file_name)
        with open(file_path, "w") as file:
            file.write(content)


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
