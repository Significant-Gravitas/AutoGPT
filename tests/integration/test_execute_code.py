import pathlib
import random
import string
import tempfile
from unittest.mock import MagicMock

import pytest

import autogpt.commands.execute_code as sut


@pytest.fixture(autouse=True)
def mock_config(mocker):
    config_mock = MagicMock(
        wraps=sut.CFG, execute_local_commands=True, workspace_path="/"
    )
    yield mocker.patch("autogpt.commands.execute_code.CFG", config_mock)


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(mock_config, random_string):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as temp_file:
        temp_file.write(str.encode(f"print('Hello {random_string}!')"))
        temp_file.flush()
        mock_config.workspace_path = pathlib.Path(temp_file.name).parent
        result = sut.execute_python_file(temp_file.name)
        assert result == f"Hello {random_string}!\n"


def test_execute_shell(random_string):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'")
    assert f"Hello {random_string}!" in result
