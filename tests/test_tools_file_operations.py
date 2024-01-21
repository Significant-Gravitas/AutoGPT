import hashlib
import os
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from pytest_mock import MockerFixture

import AFAAS.core.tools.builtins.file_operations as file_ops

# from AFAAS.configs.config import Config
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.agent.main import BaseAgent
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
)
from tests.dataset.test_tools_file import (
    file_content,
    test_directory,
    test_file,
    test_file_name,
    test_file_path,
    test_file_with_content_path,
    test_nested_file,
)

# FIXME:
# def test_log_operation_with_checksum(agent: BaseAgent):
#     file_ops.log_operation(
#         "log_test", Path("path/to/test"), agent=default_task.agent, checksum="ABCDEF"
#     )
#     with open(agent.file_manager.file_ops_log_path, "r", encoding="utf-8") as f:
#         content = f.read()
#     assert "log_test: path/to/test #ABCDEF\n" in content


def test_read_file(
    default_task: Task,
    test_file_with_content_path: Path,
    file_content,
    agent: BaseAgent,
):
    content = file_ops.read_file(
        filename=test_file_with_content_path,
        agent=default_task.agent,
        task=default_task,
    )

    assert content.replace("\r", "") == file_content


def test_read_file_not_found(default_task: Task, agent: BaseAgent):
    filename = "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        file_ops.read_file(
            filename=filename,
            agent=default_task.agent,
            task=default_task,
        )


# FIXME:NOT NotImplementedError
@pytest.mark.asyncio
async def test_write_to_file_relative_path(
    default_task: Task, test_file_name: Path, agent: BaseAgent
):
    new_content = "This is new content.\n"
    await file_ops.write_to_file(
        filename=test_file_name,
        contents=new_content,
        agent=default_task.agent,
        task=default_task,
    )

    with open(agent.workspace.get_path(test_file_name), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


# FIXME:NOT NotImplementedError
@pytest.mark.asyncio
async def test_write_to_file_absolute_path(
    test_file_path: Path, agent: BaseAgent, default_task: Task
):
    new_content = "This is new content.\n"
    await file_ops.write_to_file(
        filename=test_file_path,
        contents=new_content,
        agent=default_task.agent,
        task=default_task,
    )

    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


# FIXME:NOT NotImplementedError
@pytest.mark.asyncio
async def test_write_file_succeeds_if_content_different(
    default_task: Task,
    test_file_with_content_path: Path,
    agent: BaseAgent,
):
    new_content = "This is different content.\n"
    await file_ops.write_to_file(
        test_file_with_content_path,
        new_content,
        agent=default_task.agent,
        task=default_task,
    )


def test_list_files(
    default_task: Task,
    local_workspace: AbstractFileWorkspace,
    test_directory: Path,
    agent: BaseAgent,
):
    # Case 1: Create files A and B, search for A, and ensure we don't return A and B
    file_a = local_workspace.get_path("file_a.txt")
    file_b = local_workspace.get_path("file_b.txt")

    with open(file_a, "w") as f:
        f.write("This is file A.")

    with open(file_b, "w") as f:
        f.write("This is file B.")

    # Create a subdirectory and place a copy of file_a in it
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    with open(os.path.join(test_directory, file_a.name), "w") as f:
        f.write("This is file A in the subdirectory.")

    files = file_ops.list_folder(
        folder=str(local_workspace.root),
        agent=default_task.agent,
        task=default_task,
    )
    assert file_a.name in files
    assert file_b.name in files
    assert os.path.join(Path(test_directory).name, file_a.name) in files

    # Clean up
    os.remove(file_a)
    os.remove(file_b)
    os.remove(os.path.join(test_directory, file_a.name))
    os.rmdir(test_directory)

    # Case 2: Search for a file that does not exist and make sure we don't throw
    non_existent_file = "non_existent_file.txt"
    files = file_ops.list_folder(
        folder="",
        agent=default_task.agent,
        task=default_task,
    )
    assert non_existent_file not in files
