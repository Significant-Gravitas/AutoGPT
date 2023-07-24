import os
import random
import string
import tempfile
from typing import Generator

import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    AccessDeniedError,
    InvalidArgumentError,
    OperationNotAllowedError,
)
from autogpt.config import Config


@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"


@pytest.fixture
def python_test_file(config: Config, random_code: str) -> Generator[str, None, None]:
    temp_file = tempfile.NamedTemporaryFile(dir=config.workspace_path, suffix=".py")
    temp_file.write(str.encode(random_code))
    temp_file.flush()

    yield temp_file.name
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(python_test_file: str, random_string: str, agent: Agent):
    result: str = sut.execute_python_file(python_test_file, agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_code(random_code: str, random_string: str, agent: Agent):
    ai_name = agent.ai_config.ai_name

    result: str = sut.execute_python_code(random_code, b"test_code", agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"

    # Check that the code is stored
    destination = os.path.join(
        str(agent.config.workspace_path).encode(),
        ai_name.encode(),
        b"executed_code",
        b"test_code.py",
    )
    with open(destination, "rb") as f:
        assert f.read().decode() == random_code


def test_execute_python_code_disallows_name_arg_path_traversal(
    random_code: str, agent: Agent
):
    with pytest.raises(AccessDeniedError, match="path traversal"):
        sut.execute_python_code(random_code, name="../../test_code", agent=agent)

    # Check that the code is not stored in parent directory
    dst_with_traversal = agent.workspace.get_path("test_code.py")
    assert not dst_with_traversal.is_file(), "Path traversal by filename not prevented"


def test_execute_python_code_overwrites_file(random_code: str, agent: Agent):
    ai_name = agent.ai_config.ai_name
    destination = os.path.join(
        str(agent.config.workspace_path).encode(),
        ai_name.encode(),
        b"executed_code",
        b"test_code.py",
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "w+") as f:
        f.write("This will be overwritten")

    sut.execute_python_code(random_code, "test_code.py", agent=agent)

    # Check that the file is updated with the new code
    with open(destination) as f:
        assert f.read() == random_code


def test_execute_python_file_invalid(agent: Agent):
    with pytest.raises(InvalidArgumentError):
        sut.execute_python_file("not_python", agent)


def test_execute_python_file_not_found(agent: Agent):
    with pytest.raises(
        FileNotFoundError,
        match=r"python: can't open file '([a-zA-Z]:)?[/\\\-\w]*notexist.py': \[Errno 2\] No such file or directory",
    ):
        sut.execute_python_file("notexist.py", agent)


def test_execute_shell(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    agent.config.shell_denylist = ["echo"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    agent.config.shell_denylist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result


def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    agent.config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.config.shell_allowlist = ["cat"]

    with pytest.raises(OperationNotAllowedError, match="not allowed"):
        sut.execute_shell(f"echo 'Hello {random_string}!'", agent)


def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    agent.config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.config.shell_allowlist = ["echo"]

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
