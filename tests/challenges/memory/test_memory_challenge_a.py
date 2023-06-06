import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import run_interaction_loop
from tests.utils import requires_api_key


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_memory_challenge_a(
    memory_management_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    level_to_run: int,
) -> None:
    """
    The agent reads a file containing a task_id. Then, it reads a series of other files.
    After reading 'n' files, the agent must write the task_id into a new file.
    Args:
        memory_management_agent (Agent)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        config (Config)
        level_to_run (int)
    """

    task_id = "2314"
    create_instructions_files(memory_management_agent, level_to_run, task_id, config)

    run_interaction_loop(monkeypatch, memory_management_agent, level_to_run + 2)

    file_path = str(memory_management_agent.workspace.get_path("output.txt"))
    content = read_file(file_path, config)
    assert task_id in content, f"Expected the file to contain {task_id}"


def create_instructions_files(
    memory_management_agent: Agent,
    num_files: int,
    task_id: str,
    config: Config,
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
        file_path = str(memory_management_agent.workspace.get_path(file_name))
        write_to_file(file_path, content, config)


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
