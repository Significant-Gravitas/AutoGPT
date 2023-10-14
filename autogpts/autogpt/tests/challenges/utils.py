import contextlib
import random
import shutil
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from agbenchmark_config.benchmarks import run_specific_agent
from autogpt.file_workspace import FileWorkspace
from autogpt.logs import LogCycleHandler
from tests.challenges.schema import Task


def generate_noise(noise_size: int) -> str:
    random.seed(42)
    return "".join(
        random.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            k=noise_size,
        )
    )


def setup_mock_input(monkeypatch: pytest.MonkeyPatch, cycle_count: int) -> None:
    """
    Sets up the mock input for testing.

    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    :param cycle_count: The number of cycles to mock.
    """
    input_sequence = ["y"] * (cycle_count) + ["EXIT"]

    async def input_generator() -> AsyncIterator[str]:
        """
        Creates a generator that yields input strings from the given sequence.
        """
        for input in input_sequence:
            yield input

    gen = input_generator()
    monkeypatch.setattr(
        "autogpt.app.utils.session.prompt_async", lambda _, **kwargs: anext(gen)
    )


def setup_mock_log_cycle_agent_name(
    monkeypatch: pytest.MonkeyPatch, challenge_name: str, level_to_run: int
) -> None:
    def mock_get_agent_short_name(*args: Any, **kwargs: Any) -> str:
        return f"{challenge_name}_level_{level_to_run}"

    monkeypatch.setattr(
        LogCycleHandler, "get_agent_short_name", mock_get_agent_short_name
    )


def get_workspace_path(workspace: FileWorkspace, file_name: str) -> str:
    return str(workspace.get_path(file_name))


def copy_file_into_workspace(
    workspace: FileWorkspace, directory_path: Path, file_path: str
) -> None:
    workspace_code_file_path = get_workspace_path(workspace, file_path)
    code_file_path = directory_path / file_path
    shutil.copy(code_file_path, workspace_code_file_path)


def run_challenge(
    challenge_name: str,
    level_to_run: int,
    monkeypatch: pytest.MonkeyPatch,
    user_input: str,
    cycle_count: int,
) -> None:
    setup_mock_input(monkeypatch, cycle_count)
    setup_mock_log_cycle_agent_name(monkeypatch, challenge_name, level_to_run)
    task = Task(user_input=user_input)
    with contextlib.suppress(SystemExit):
        run_specific_agent(task.user_input)
