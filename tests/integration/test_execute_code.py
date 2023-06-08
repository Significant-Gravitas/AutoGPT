import random
import string
import tempfile
from typing import Callable

import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.config import Config


@pytest.fixture
def python_test_file(config: Config, random_string) -> Callable:
    temp_file = tempfile.NamedTemporaryFile(dir=config.workspace_path, suffix=".py")
    temp_file.write(str.encode(f"print('Hello {random_string}!')"))
    temp_file.flush()

    yield temp_file.name
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(python_test_file: str, random_string: str, config):
    result: str = sut.execute_python_file(python_test_file, config)
    assert result.replace("\r", "") == f"Hello {random_string}!\n"


def test_execute_python_file_invalid(config: Config):
    assert all(
        s in sut.execute_python_file("not_python", config).lower()
        for s in ["error:", "invalid", ".py"]
    )


def test_execute_python_file_not_found(config: Config):
    assert all(
        s in sut.execute_python_file("notexist.py", config).lower()
        for s in [
            "python: can't open file 'notexist.py'",
            "[errno 2] no such file or directory",
        ]
    )


def test_execute_shell(random_string: str, config: Config):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert f"Hello {random_string}!" in result


def test_execute_shell_local_commands_not_allowed(random_string: str, config: Config):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert f"Hello {random_string}!" in result


def test_execute_shell_denylist_should_deny(config: Config, random_string: str):
    config.shell_denylist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert "Error:" in result and "not allowed" in result


def test_execute_shell_denylist_should_allow(config: Config, random_string: str):
    config.shell_denylist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert "Hello" in result and random_string in result
    assert "Error" not in result


def test_execute_shell_allowlist_should_deny(config: Config, random_string: str):
    config.shell_command_control = sut.ALLOWLIST_CONTROL
    config.shell_allowlist = ["cat"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert "Error:" in result and "not allowed" in result


def test_execute_shell_allowlist_should_allow(config: Config, random_string: str):
    config.shell_command_control = sut.ALLOWLIST_CONTROL
    config.shell_allowlist = ["echo"]

    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert "Hello" in result and random_string in result
    assert "Error" not in result
