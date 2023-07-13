import pytest
from pytest_mock import MockerFixture

from autogpt.config import Config
from autogpt.workspace import Workspace
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path, run_challenge

OUTPUT_LOCATION = "output.txt"

USER_INPUT = "Use the command read_file to read the instructions_1.txt file\nFollow the instructions in the instructions_1.txt file"


@challenge()
def test_memory_challenge_a(
    config: Config,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    workspace: Workspace,
    patched_make_workspace: pytest.fixture,
) -> None:
    """
    The agent reads a file containing a task_id. Then, it reads a series of other files.
    After reading 'n' files, the agent must write the task_id into a new file.
    Args:
        workspace (Workspace)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    task_id = "2314"
    create_instructions_files(workspace, level_to_run, task_id)

    run_challenge(
        challenge_name, level_to_run, monkeypatch, USER_INPUT, level_to_run + 2
    )

    file_path = get_workspace_path(workspace, OUTPUT_LOCATION)
    with open(file_path, "r") as file:
        content = file.read()
    assert task_id in content, f"Expected the file to contain {task_id}"


def create_instructions_files(
    workspace: Workspace,
    num_files: int,
    task_id: str,
    base_filename: str = "instructions_",
) -> None:
    """
    Creates a series of instructions files for the memory challenge.
    Args:
        workspace (Workspace)
        num_files (int)
        task_id (str)
        base_filename (str, optional)
    """
    for i in range(1, num_files + 1):
        content = generate_content(i, task_id, base_filename, num_files)
        file_name = f"{base_filename}{i}.txt"
        file_path = get_workspace_path(workspace, file_name)
        with open(file_path, "w") as file:
            file.write(content)


def generate_content(
    index: int, task_id: str, base_filename: str, num_files: int
) -> str:
    """
    Args:
        index: int
        task_id: str
        base_filename: str
        num_files: int

    Returns: str
    """
    if index == 1:
        return (
            f"This task_id is {task_id}\nRead the file {base_filename}{index + 1}.txt"
        )
    if index != num_files:
        return f"Read the file {base_filename}{index + 1}.txt"
    return "Write the task_id into the file output.txt\nShutdown"
