import random
import string
import tempfile

import pytest
from pytest_mock import MockerFixture

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.config import Config


@pytest.fixture(autouse=True)
def config_allow_execute(config: Config, mocker: MockerFixture):
    yield mocker.patch.object(config, "execute_local_commands", True)


@pytest.fixture
def random_string():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(10))


def test_execute_python_file(random_string, workspace_root):
    with tempfile.NamedTemporaryFile(
        dir=str(workspace_root), suffix=".py"
    ) as temp_file:
        temp_file.write(str.encode(f"print('Hello {random_string}!')"))
        temp_file.flush()
        result = sut.execute_python_file(temp_file.name)
        assert result == f"Hello {random_string}!\n"


def test_execute_shell(random_string):
    result = sut.execute_shell(f"echo 'Hello {random_string}!'")
    assert f"Hello {random_string}!" in result
