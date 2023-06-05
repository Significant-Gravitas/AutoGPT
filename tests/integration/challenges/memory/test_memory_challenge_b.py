import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.commands.file_operations import read_file, write_to_file
from autogpt.config import Config
from tests.integration.challenges.challenge_decorator.challenge_decorator import (
    challenge,
)
from tests.integration.challenges.utils import generate_noise, run_interaction_loop
from tests.utils import requires_api_key

NOISE = 1000


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
@challenge
def test_memory_challenge_b(
    memory_management_agent: Agent,
    patched_api_requestor: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    level_to_run: int,
) -> None:
    """
    The agent reads a series of files, each containing a task_id and noise. After reading 'n' files,
    the agent must write all the task_ids into a new file, filtering out the noise.

    Args:
        memory_management_agent (Agent)
        patched_api_requestor (MockerFixture)
        monkeypatch (pytest.MonkeyPatch)
        level_to_run (int)
    """
    task_ids = [str(i * 1111) for i in range(1, level_to_run + 1)]
    create_instructions_files(memory_management_agent, level_to_run, task_ids, config)

    run_interaction_loop(monkeypatch, memory_management_agent, level_to_run + 2)

    file_path = str(memory_management_agent.workspace.get_path("output.txt"))
    content = read_file(file_path, config)
    for task_id in task_ids:
        assert task_id in content, f"Expected the file to contain {task_id}"


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
