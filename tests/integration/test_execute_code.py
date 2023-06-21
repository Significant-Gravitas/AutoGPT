import os
import random
import string
import tempfile

import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.agent.agent import Agent
from autogpt.config import Config


@pytest.fixture
def random_code(random_string) -> str:
    return f"print('Hello {random_string}!')"


@pytest.fixture
def python_test_file(config: Config, random_code: str) -> str:
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
    ai_name = agent.ai_name

    result: str = sut.execute_python_code(random_code, "test_code", agent=agent)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"

    # Check that the code is stored
    destination = os.path.join(
        agent.config.workspace_path, ai_name, "executed_code", "test_code.py"
    )
    with open(destination) as f:
        assert f.read() == random_code


def test_execute_python_code_disallows_name_arg_path_traversal(
    random_code: str, agent: Agent
):
    result: str = sut.execute_python_code(
        random_code, name="../../test_code", agent=agent
    )
    assert "Error:" in result, "Path traversal in 'name' argument does not return error"
    assert "path traversal" in result.lower()

    # Check that the code is not stored in parent directory
    dst_with_traversal = agent.workspace.get_path("test_code.py")
    assert not dst_with_traversal.is_file(), "Path traversal by filename not prevented"


def test_execute_python_code_overwrites_file(random_code: str, agent: Agent):
    ai_name = agent.ai_name
    destination = os.path.join(
        agent.config.workspace_path, ai_name, "executed_code", "test_code.py"
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "w+") as f:
        f.write("This will be overwritten")

    sut.execute_python_code(random_code, "test_code.py", agent=agent)

    # Check that the file is updated with the new code
    with open(destination) as f:
        assert f.read() == random_code


def test_execute_python_file_invalid(agent: Agent):
    assert all(
        s in sut.execute_python_file("not_python", agent).lower()
        for s in ["error:", "invalid", ".py"]
    )


def test_execute_python_file_not_found(agent: Agent):
    assert all(
        s in sut.execute_python_file("notexist.py", agent).lower()
        for s in [
            "python: can't open file 'notexist.py'",
            "[errno 2] no such file or directory",
        ]
    )


def test_execute_shell(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(random_string: str, agent: Agent):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert f"Hello {random_string}!" in result


def test_execute_shell_denylist_should_deny(agent: Agent, random_string: str):
    agent.config.shell_denylist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Error:" in result and "not allowed" in result


def test_execute_shell_denylist_should_allow(agent: Agent, random_string: str):
    agent.config.shell_denylist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result
    assert "Error" not in result


def test_execute_shell_allowlist_should_deny(agent: Agent, random_string: str):
    agent.config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.config.shell_allowlist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Error:" in result and "not allowed" in result


def test_execute_shell_allowlist_should_allow(agent: Agent, random_string: str):
    agent.config.shell_command_control = sut.ALLOWLIST_CONTROL
    agent.config.shell_allowlist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", agent)
    assert "Hello" in result and random_string in result
    assert "Error" not in result
