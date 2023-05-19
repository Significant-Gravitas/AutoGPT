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
def python_test_file_args(config: Config):
    temp_file = tempfile.NamedTemporaryFile(dir=config.workspace_path, suffix=".py")
    temp_file.write(str.encode(f"import sys\nprint(sys.argv[1], sys.argv[2])"))
    temp_file.flush()

    yield temp_file.name
    temp_file.close()


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(python_test_file: str, random_string: str):
    result = sut.execute_python_file(python_test_file)
    assert result == f"Hello {random_string}!\n"


def test_execute_python_file_args(python_test_file_args: str, random_string: str):
    random_args = random_string + " " + random_string
    result = sut.execute_python_file(python_test_file_args, random_args)
    assert result == f"{random_args}\n"


def test_execute_python_file_invalid():
    assert all(
        s in sut.execute_python_file("not_python").lower()
        for s in ["error:", "invalid", ".py"]
    )
    assert all(
        s in sut.execute_python_file("notexist.py").lower()
        for s in ["error:", "does not exist"]
    )


def test_execute_shell(config_allow_execute, random_string):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'")
    assert f"Hello {random_string}!" in result
