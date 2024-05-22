import os
from pathlib import Path

import pytest
from forge.file_storage import FileStorage

from autogpt.agents.agent import Agent


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


@pytest.fixture
def file_manager_component(agent: Agent):
    return agent.file_manager


@pytest.fixture()
def test_file_name():
    return Path("test_file.txt")


@pytest.fixture
def test_file_path(test_file_name: Path, storage: FileStorage):
    return storage.get_path(test_file_name)


@pytest.fixture()
def test_directory(storage: FileStorage):
    return storage.get_path("test_directory")


@pytest.fixture()
def test_nested_file(storage: FileStorage):
    return storage.get_path("nested/test_file.txt")


@pytest.mark.asyncio
async def test_read_file(
    test_file_path: Path,
    file_content,
    file_manager_component,
    agent: Agent,
):
    await agent.file_manager.workspace.write_file(test_file_path.name, file_content)
    content = file_manager_component.read_file(test_file_path.name)
    assert content.replace("\r", "") == file_content


def test_read_file_not_found(file_manager_component):
    filename = "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        file_manager_component.read_file(filename)


@pytest.mark.asyncio
async def test_write_to_file_relative_path(
    test_file_name: Path, file_manager_component, agent: Agent
):
    new_content = "This is new content.\n"
    await file_manager_component.write_to_file(test_file_name, new_content)
    with open(
        agent.file_manager.workspace.get_path(test_file_name), "r", encoding="utf-8"
    ) as f:
        content = f.read()
    assert content == new_content


@pytest.mark.asyncio
async def test_write_to_file_absolute_path(
    test_file_path: Path, file_manager_component
):
    new_content = "This is new content.\n"
    await file_manager_component.write_to_file(test_file_path, new_content)
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


@pytest.mark.asyncio
async def test_list_files(file_manager_component, agent: Agent):
    # Create files A and B
    file_a_name = "file_a.txt"
    file_b_name = "file_b.txt"
    test_directory = Path("test_directory")

    await agent.file_manager.workspace.write_file(file_a_name, "This is file A.")
    await agent.file_manager.workspace.write_file(file_b_name, "This is file B.")

    # Create a subdirectory and place a copy of file_a in it
    agent.file_manager.workspace.make_dir(test_directory)
    await agent.file_manager.workspace.write_file(
        test_directory / file_a_name, "This is file A in the subdirectory."
    )

    files = file_manager_component.list_folder(".")
    assert file_a_name in files
    assert file_b_name in files
    assert os.path.join(test_directory, file_a_name) in files

    # Clean up
    agent.file_manager.workspace.delete_file(file_a_name)
    agent.file_manager.workspace.delete_file(file_b_name)
    agent.file_manager.workspace.delete_file(test_directory / file_a_name)
    agent.file_manager.workspace.delete_dir(test_directory)

    # Case 2: Search for a file that does not exist and make sure we don't throw
    non_existent_file = "non_existent_file.txt"
    files = file_manager_component.list_folder("")
    assert non_existent_file not in files
