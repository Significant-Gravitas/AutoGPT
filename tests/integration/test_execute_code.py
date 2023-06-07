import random
import string
import tempfile
from typing import Callable

import pytest
from pytest_mock import MockerFixture

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.config import Config


@pytest.fixture
def config_allow_execute(config: Config, mocker: MockerFixture) -> Callable:
    yield mocker.patch.object(config, "execute_local_commands", True)


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


def test_execute_shell(config_allow_execute: bool, random_string: str, config: Config):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert f"Hello {random_string}!" in result


def test_execute_shell_deny_command(
    python_test_file: str, config_allow_execute: bool, config: Config
):
    config.deny_commands = ["echo"]
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert "Error:" in result and "not allowed" in result
