import pytest
from pytest_mock import MockerFixture

from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import generate_noise, get_workspace_path, run_challenge

NOISE = 1000
OUTPUT_LOCATION = "output.txt"
USER_INPUT = "Use the command read_file to read the instructions_1.txt file\nFollow the instructions in the instructions_1.txt file"


@challenge()
def test_memory_challenge_b(
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    The agent reads a series of files, each containing a task_id and noise. After reading 'n' files,
    the agent must write all the task_ids into a new file, filtering out the noise.

    Args:
        workspace (Workspace)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    task_ids = [str(i * 1111) for i in range(1, level_to_run + 1)]
    create_instructions_files(workspace, level_to_run, task_ids)

    run_challenge(
        challenge_name, level_to_run, monkeypatch, USER_INPUT, level_to_run + 2
    )

    file_path = get_workspace_path(workspace, OUTPUT_LOCATION)
    with open(file_path, "r") as file:
        content = file.read()
    for task_id in task_ids:
        assert task_id in content, f"Expected the file to contain {task_id}"


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


def generate_content(index: int, task_ids: list, base_filename: str, level: int) -> str:
    """
    Args:
        index: int
        task_ids: list
        base_filename: str
        num_files: int

    Returns: str
    """
    task_id = task_ids[index - 1]
    noise = generate_noise(NOISE)
    if index != level:
        if level == 1:
            return f"{noise}\nThe current task_id is {task_id}.\n{noise}\nWrite all the task_ids into the file output.txt. The file has not been created yet. After that, use the task_complete command."
        return f"{noise}\nThe current task_id is {task_id}.\n{noise}\nRead the file {base_filename}{index + 1}.txt using the read_file command."
    return f"{noise}\nThis task_id is {task_id}\n{noise}\nWrite all the task_ids into the file output.txt. The file has not been created yet. After that, use the task_complete command.\n"
