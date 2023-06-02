import random
import string
import tempfile

import pytest
from pytest_mock import MockerFixture

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.config import Config


@pytest.fixture
def config_allow_execute(config: Config, mocker: MockerFixture):
    yield mocker.patch.object(config, "execute_local_commands", True)


@pytest.fixture
def python_test_file(config: Config, random_string):
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


def test_execute_python_file_invalid(config):
    assert all(
        s in sut.execute_python_file("not_python", config).lower()
        for s in ["error:", "invalid", ".py"]
    )
    assert all(
        s in sut.execute_python_file("notexist.py", config).lower()
        for s in ["error:", "does not exist"]
    )


def test_execute_shell(config_allow_execute, random_string, config):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'", config)
    assert f"Hello {random_string}!" in result
