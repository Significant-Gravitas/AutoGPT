import contextlib
import random
import shutil
from pathlib import Path
from typing import Generator

import pytest

from autogpt.agent import Agent


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

    def input_generator() -> Generator[str, None, None]:
        """
        Creates a generator that yields input strings from the given sequence.
        """
        yield from input_sequence

    gen = input_generator()
    monkeypatch.setattr("autogpt.utils.session.prompt", lambda _: next(gen))


def run_interaction_loop(
    monkeypatch: pytest.MonkeyPatch, agent: Agent, cycle_count: int
) -> None:
    setup_mock_input(monkeypatch, cycle_count)
    with contextlib.suppress(SystemExit):
        agent.start_interaction_loop()


def get_workspace_path(agent: Agent, file_name: str) -> str:
    return str(agent.workspace.get_path(file_name))


def copy_file_into_workspace(
    agent: Agent, directory_path: Path, file_path: str
) -> None:
    workspace_code_file_path = get_workspace_path(agent, file_path)
    code_file_path = directory_path / file_path
    shutil.copy(code_file_path, workspace_code_file_path)
