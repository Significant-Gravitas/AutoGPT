import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.utils import get_workspace_path_from_agent, run_interaction_loop

OUTPUT_LOCATION = "output.txt"


@challenge()
def test_memory_challenge_a(
    memory_management_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
) -> None:
    """
    The agent reads a file containing a task_id. Then, it reads a series of other files.
    After reading 'n' files, the agent must write the task_id into a new file.
    Args:
        memory_management_agent (Agent)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    task_id = "2314"
    create_instructions_files(memory_management_agent, level_to_run, task_id)

    run_interaction_loop(
        monkeypatch,
        memory_management_agent,
        level_to_run + 2,
        challenge_name,
        level_to_run,
    )

    file_path = get_workspace_path_from_agent(memory_management_agent, OUTPUT_LOCATION)
    content = read_file(file_path, memory_management_agent)
    assert task_id in content, f"Expected the file to contain {task_id}"


def create_instructions_files(
    memory_management_agent: Agent,
    num_files: int,
    task_id: str,
    base_filename: str = "instructions_",
) -> None:
    """
    Creates a series of instructions files for the memory challenge.
    Args:
        memory_management_agent (Agent)
        num_files (int)
        task_id (str)
        base_filename (str, optional)
    """
    for i in range(1, num_files + 1):
        content = generate_content(i, task_id, base_filename, num_files)
        file_name = f"{base_filename}{i}.txt"
        file_path = get_workspace_path_from_agent(memory_management_agent, file_name)
        write_to_file(file_path, content, memory_management_agent)


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
