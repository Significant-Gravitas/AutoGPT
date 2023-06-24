import pytest

from autogpt.workspace import Workspace
from benchmarks import run_task
from tests.challenges.challenge_decorator.challenge_decorator import challenge
from tests.challenges.schema import Task
from tests.challenges.utils import (
    get_workspace_path,
    setup_mock_input,
    setup_mock_log_cycle_agent_name,
)

CYCLE_COUNT_PER_LEVEL = [1, 1]
EXPECTED_OUTPUTS_PER_LEVEL = [
    {"hello_world.txt": ["Hello World"]},
    {"hello_world_1.txt": ["Hello World"], "hello_world_2.txt": ["Hello World"]},
]
USER_INPUTS = [
    "Write 'Hello World' into a file named \"hello_world.txt\".",
    'Write \'Hello World\' into 2 files named "hello_world_1.txt"and "hello_world_2.txt".',
]


@challenge()
def test_write_file(
    workspace: Workspace,
    patched_api_requestor: None,
    monkeypatch: pytest.MonkeyPatch,
    level_to_run: int,
    challenge_name: str,
    patched_make_workspace: pytest.fixture,
) -> None:
    user_input = USER_INPUTS[level_to_run - 1]
    setup_mock_input(monkeypatch, CYCLE_COUNT_PER_LEVEL[level_to_run - 1])
    setup_mock_log_cycle_agent_name(monkeypatch, challenge_name, level_to_run)
    task = Task(user_input=user_input)
    run_task(task)

    expected_outputs = EXPECTED_OUTPUTS_PER_LEVEL[level_to_run - 1]

    for file_name, expected_lines in expected_outputs.items():
        file_path = get_workspace_path(workspace, file_name)
        with open(file_path, "r") as file:
            content = file.read()

        for expected_line in expected_lines:
            assert (
                expected_line in content
            ), f"Expected '{expected_line}' in file {file_name}, but it was not found"
