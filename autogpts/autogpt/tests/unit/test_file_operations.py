import os
import re
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import autogpt.commands.file_operations as file_ops
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import DuplicateOperationError
from autogpt.config import Config
from autogpt.file_storage import FileStorage
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding


@pytest.fixture()
def file_content() -> str:
    return "This is a test file.\n"


@pytest.fixture()
def mock_MemoryItem_from_text(
    mocker: MockerFixture, mock_embedding: Embedding, config: Config
) -> None:
    mocker.patch.object(
        file_ops.MemoryItemFactory,
        "from_text",
        new=lambda content, source_type, config, metadata: MemoryItem(
            raw_content=content,
            summary=f"Summary of content '{content}'",
            chunk_summaries=[f"Summary of content '{content}'"],
            chunks=[content],
            e_summary=mock_embedding,
            e_chunks=[mock_embedding],
            metadata=metadata | {"source_type": source_type},
        ),
    )


@pytest.fixture()
def test_file_name() -> Path:
    return Path("test_file.txt")


@pytest.fixture()
def test_file_path(test_file_name: Path, storage: FileStorage) -> Path:
    return storage.get_path(test_file_name)


@pytest.fixture()
def test_directory(storage: FileStorage) -> Path:
    return storage.get_path("test_directory")


@pytest.fixture()
def test_nested_file(storage: FileStorage) -> Path:
    return storage.get_path("nested/test_file.txt")


def test_file_operations_log() -> None:
    all_logs = (
        "File Operation Logger\n"
        "write: path/to/file1.txt #checksum1\n"
        "write: path/to/file2.txt #checksum2\n"
        "write: path/to/file3.txt #checksum3\n"
        "append: path/to/file2.txt #checksum4\n"
        "delete: path/to/file3.txt\n"
    )
    logs: list[str] = all_logs.split("\n")

    expected = [
        (file_ops.Operations.WRITE, "path/to/file1.txt", "checksum1"),
        (file_ops.Operations.WRITE, "path/to/file2.txt", "checksum2"),
        (file_ops.Operations.WRITE, "path/to/file3.txt", "checksum3"),
        (file_ops.Operations.APPEND, "path/to/file2.txt", "checksum4"),
        (file_ops.Operations.DELETE, "path/to/file3.txt", None),
    ]
    assert list(file_ops.operations_from_log(logs)) == expected


def test_is_duplicate_operation(agent: Agent, mocker: MockerFixture) -> None:
    # Prepare a fake state dictionary for the function to use
    state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum2",
    }
    mocker.patch.object(file_ops, "file_operations_state", lambda _: state)

    # Test cases with write operations
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.WRITE, Path("path/to/file1.txt"), agent, "checksum1"
        )
        is True
    )
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.WRITE, Path("path/to/file1.txt"), agent, "checksum2"
        )
        is False
    )
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.WRITE, Path("path/to/file3.txt"), agent, "checksum3"
        )
        is False
    )
    # Test cases with append operations
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.APPEND, Path("path/to/file1.txt"), agent, "checksum1"
        )
        is False
    )
    # Test cases with delete operations
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.DELETE, Path("path/to/file1.txt"), agent
        )
        is False
    )
    assert (
        file_ops.is_duplicate_operation(
            file_ops.Operations.DELETE, Path("path/to/file3.txt"), agent
        )
        is True
    )


# Test logging a file operation
@pytest.mark.asyncio
async def test_log_operation(agent: Agent) -> None:
    await file_ops.log_operation(
        file_ops.Operations.APPEND, Path("path/to/test"), agent=agent
    )
    log_entry = agent.get_file_operation_lines()[-1]
    assert "append: path/to/test" in log_entry


def test_text_checksum(file_content: str) -> None:
    checksum = file_ops.text_checksum(file_content)
    different_checksum = file_ops.text_checksum("other content")
    assert re.match(r"^[a-fA-F0-9]+$", checksum) is not None
    assert checksum != different_checksum


@pytest.mark.asyncio
async def test_log_operation_with_checksum(agent: Agent) -> None:
    await file_ops.log_operation(
        file_ops.Operations.WRITE, Path("path/to/test"), agent=agent, checksum="ABCDEF"
    )
    log_entry = agent.get_file_operation_lines()[-1]
    assert "write: path/to/test #ABCDEF" in log_entry


@pytest.mark.asyncio
async def test_read_file(
    mock_MemoryItem_from_text,
    test_file_path: Path,
    file_content: str,
    agent: Agent,
):
    await agent.workspace.write_file(test_file_path.name, file_content)
    await file_ops.log_operation(
        file_ops.Operations.WRITE,
        test_file_path.name,
        agent,
        file_ops.text_checksum(file_content),
    )
    content = file_ops.read_file(test_file_path.name, agent=agent)
    assert content.replace("\r", "") == file_content


def test_read_file_not_found(agent: Agent) -> None:
    filename = "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        file_ops.read_file(filename, agent=agent)


@pytest.mark.asyncio
async def test_write_to_file_relative_path(test_file_name: Path, agent: Agent) -> None:
    new_content = "This is new content.\n"
    await file_ops.write_to_file(test_file_name, new_content, "overwrite", agent=agent)
    with open(agent.workspace.get_path(test_file_name), "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


@pytest.mark.asyncio
async def test_write_to_file_absolute_path(test_file_path: Path, agent: Agent) -> None:
    new_content = "This is new content.\n"
    await file_ops.write_to_file(test_file_path, new_content, "overwrite", agent=agent)
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


@pytest.mark.asyncio
async def test_write_file_logs_checksum(test_file_name: Path, agent: Agent) -> None:
    new_content = "This is new content.\n"
    new_checksum = file_ops.text_checksum(new_content)
    await file_ops.write_to_file(test_file_name, new_content, "overwrite", agent=agent)
    log_entry = agent.get_file_operation_lines()[-1]
    assert log_entry == f"write: {test_file_name} #{new_checksum}"


@pytest.mark.asyncio
async def test_write_file_fails_if_content_exists(
    test_file_name: Path, agent: Agent
) -> None:
    new_content = "This is new content.\n"
    await file_ops.log_operation(
        file_ops.Operations.WRITE,
        test_file_name,
        agent=agent,
        checksum=file_ops.text_checksum(new_content),
    )
    with pytest.raises(DuplicateOperationError):
        await file_ops.write_to_file(
            test_file_name, new_content, "overwrite", agent=agent
        )


@pytest.mark.asyncio
async def test_write_file_succeeds_if_content_different(
    test_file_path: Path, file_content: str, agent: Agent
) -> None:
    await agent.workspace.write_file(test_file_path.name, file_content)
    await file_ops.log_operation(
        file_ops.Operations.WRITE,
        Path(test_file_path.name),
        agent,
        file_ops.text_checksum(file_content),
    )
    new_content = "This is different content.\n"
    await file_ops.write_to_file(
        test_file_path.name, new_content, "overwrite", agent=agent
    )


@pytest.mark.asyncio
async def test_list_files(agent: Agent) -> None:
    # Create files A and B
    file_a_name = "file_a.txt"
    file_b_name = "file_b.txt"
    test_directory = Path("test_directory")

    await agent.workspace.write_file(file_a_name, "This is file A.")
    await agent.workspace.write_file(file_b_name, "This is file B.")

    # Create a subdirectory and place a copy of file_a in it
    agent.workspace.make_dir(test_directory)
    await agent.workspace.write_file(
        test_directory / file_a_name, "This is file A in the subdirectory."
    )
    files = file_ops.list_folder(".", agent=agent)
    assert file_a_name in files
    assert file_b_name in files
    assert os.path.join(test_directory, file_a_name) in files

    # Clean up
    agent.workspace.delete_file(file_a_name)
    agent.workspace.delete_file(file_b_name)
    agent.workspace.delete_file(test_directory / file_a_name)
    agent.workspace.delete_dir(test_directory)

    # Case 2: Search for a file that does not exist and make sure we don't throw
    non_existent_file = "non_existent_file.txt"
    files = file_ops.list_folder("", agent=agent)
    assert non_existent_file not in files
