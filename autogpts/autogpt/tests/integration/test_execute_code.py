import random
import string
import tempfile
from pathlib import Path

import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    InvalidArgumentError,
    OperationNotAllowedError,
)
from autogpt.config import Config


@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"


@pytest.fixture
def python_test_file(agent: Agent, random_code: str):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode(random_code))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


@pytest.fixture
def python_test_args_file(agent: Agent):
    temp_file = tempfile.NamedTemporaryFile(dir=agent.workspace.root, suffix=".py")
    temp_file.write(str.encode("import sys\nprint(sys.argv[1], sys.argv[2])"))
    temp_file.flush()

    yield Path(temp_file.name)
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(python_test_file: Path, random_string: str, agent: Agent):
    if not (sut.is_docker_available() or sut.we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    result: str = sut.execute_python_file(python_test_file, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_args(
    python_test_args_file: Path, random_string: str, agent: Agent
):
    if not (sut.is_docker_available() or sut.we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    random_args = [random_string] * 2
    random_args_string = " ".join(random_args)
    result = sut.execute_python_file(
        python_test_args_file, args=random_args, agent=agent
    )
    assert result == f"{random_args_string}\n"


def test_execute_python_code(random_code: str, random_string: str, agent: Agent):
    if not (sut.is_docker_available() or sut.we_are_running_in_a_docker_container()):
        pytest.skip("Docker is not available")

    result: str = sut.execute_python_code(random_code, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_invalid(agent: Agent):
    with pytest.raises(InvalidArgumentError):
        sut.execute_python_file(Path("not_python.txt"), agent)


def test_execute_python_file_not_found(agent: Agent):
    with pytest.raises(
        FileNotFoundError,
        match=r"python: can't open file '([a-zA-Z]:)?[/\\\-\w]*notexist.py': "
        r"\[Errno 2\] No such file or directory",
    ):
        sut.execute_python_file(Path("notexist.py"), agent)


def test_execute_shell(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    agent.legacy_config.shell_denylist = ["echo"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    agent.legacy_config.shell_denylist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result


def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["cat"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    agent.legacy_config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.legacy_config.shell_allowlist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result


"""
Tests for the InteractiveShellCommands class.
"""
from unittest.mock import patch


def test_ask_user() -> None:
    """Test that the ask_user method returns the expected responses."""
    prompts = ["Question 1: ", "Question 2: ", "Question 3: "]
    expected_responses = ["Answer 1", "Answer 2", "Answer 3"]
    with patch("inputimeout.inputimeout", side_effect=expected_responses):
        responses = sut.ask_user(prompts)

    assert (
        responses == expected_responses
    ), f"Expected {expected_responses} but got {responses}"


def test_ask_user_timeout() -> None:
    """Test that the ask_user method returns the expected responses when a timeout occurs."""
    prompts = ["Prompt 1:"]
    timeout = 900

    from inputimeout import TimeoutOccurred

    with patch("inputimeout.inputimeout", side_effect=TimeoutOccurred):
        responses = sut.ask_user(prompts, timeout)

    assert responses == [f"Timed out after {timeout} seconds."]
