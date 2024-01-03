from io import TextIOWrapper
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from pytest_mock import MockerFixture

from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.agent.main import BaseAgent
from tests.dataset.plan_familly_dinner import (
    Task,
    plan_familly_dinner,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)


@pytest.fixture()
def test_file_name():
    return Path("test_file.txt")


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


@pytest.fixture
def test_file_path(test_file_name: Path, local_workspace: AbstractFileWorkspace):
    return local_workspace.get_path(test_file_name)


@pytest.fixture()
def test_file(test_file_path: Path):
    file = open(test_file_path, "w")
    yield file
    if not file.closed:
        file.close()


@pytest.fixture()
def test_nested_file(local_workspace: AbstractFileWorkspace):
    return local_workspace.get_path("nested/test_file.txt")


@pytest.fixture()
def test_file_with_content_path(
    task_ready_no_predecessors_or_subtasks: Task,
    test_file: TextIOWrapper,
    file_content,
    agent: BaseAgent,
):
    test_file.write(file_content)

    test_file.close()
    # file_ops.log_operation(
    #    operation= "write", file_path=Path(test_file.name), agent= agent, checksum= file_ops.text_checksum(text= file_content )
    # )
    return Path(test_file.name)


@pytest.fixture()
def test_directory(local_workspace: AbstractFileWorkspace):
    return local_workspace.get_path("test_directory")
