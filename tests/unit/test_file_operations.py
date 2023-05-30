"""
This set of unit tests is designed to test the file operations that autoGPT has access to.
"""

import hashlib
import os
import re
from io import TextIOWrapper
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import autogpt.commands.file_operations as file_ops
from autogpt.config import Config
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding
from autogpt.utils import readable_file_size
from autogpt.workspace import Workspace


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


@pytest.fixture()
def mock_MemoryItem_from_text(mocker: MockerFixture, mock_embedding: Embedding):
    mocker.patch.object(
        file_ops.MemoryItem,
        "from_text",
        new=lambda content, source_type, metadata: MemoryItem(
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
def test_file_path(config, workspace: Workspace):
    return workspace.get_path("test_file.txt")


@pytest.fixture()
def test_file(test_file_path: Path):
    file = open(test_file_path, "w")
    yield file
    if not file.closed:
        file.close()


@pytest.fixture()
def test_file_with_content_path(test_file: TextIOWrapper, file_content, config):
    test_file.write(file_content)
    test_file.close()
    file_ops.log_operation(
        "write", test_file.name, config, file_ops.text_checksum(file_content)
    )
    return Path(test_file.name)


@pytest.fixture()
def test_directory(config, workspace: Workspace):
    return workspace.get_path("test_directory")


@pytest.fixture()
def test_nested_file(config, workspace: Workspace):
    return workspace.get_path("nested/test_file.txt")


def test_file_operations_log(test_file: TextIOWrapper):
    log_file_content = (
        "File Operation Logger\n"
        "write: path/to/file1.txt #checksum1\n"
        "write: path/to/file2.txt #checksum2\n"
        "write: path/to/file3.txt #checksum3\n"
        "append: path/to/file2.txt #checksum4\n"
        "delete: path/to/file3.txt\n"
    )
    test_file.write(log_file_content)
    test_file.close()

    expected = [
        ("write", "path/to/file1.txt", "checksum1"),
        ("write", "path/to/file2.txt", "checksum2"),
        ("write", "path/to/file3.txt", "checksum3"),
        ("append", "path/to/file2.txt", "checksum4"),
        ("delete", "path/to/file3.txt", None),
    ]
    assert list(file_ops.operations_from_log(test_file.name)) == expected


def test_file_operations_state(test_file: TextIOWrapper):
    # Prepare a fake log file
    log_file_content = (
        "File Operation Logger\n"
        "write: path/to/file1.txt #checksum1\n"
        "write: path/to/file2.txt #checksum2\n"
        "write: path/to/file3.txt #checksum3\n"
        "append: path/to/file2.txt #checksum4\n"
        "delete: path/to/file3.txt\n"
    )
    test_file.write(log_file_content)
    test_file.close()

    # Call the function and check the returned dictionary
    expected_state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum4",
    }
    assert file_ops.file_operations_state(test_file.name) == expected_state


def test_is_duplicate_operation(config: Config, mocker: MockerFixture):
    # Prepare a fake state dictionary for the function to use
    state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum2",
    }
    mocker.patch.object(file_ops, "file_operations_state", lambda _: state)

    # Test cases with write operations
    assert (
        file_ops.is_duplicate_operation(
            "write", "path/to/file1.txt", config, "checksum1"
        )
        is True
    )
    assert (
        file_ops.is_duplicate_operation(
            "write", "path/to/file1.txt", config, "checksum2"
        )
        is False
    )
    assert (
        file_ops.is_duplicate_operation(
            "write", "path/to/file3.txt", config, "checksum3"
        )
        is False
    )
    # Test cases with append operations
    assert (
        file_ops.is_duplicate_operation(
            "append", "path/to/file1.txt", config, "checksum1"
        )
        is False
    )
    # Test cases with delete operations
    assert (
        file_ops.is_duplicate_operation("delete", "path/to/file1.txt", config) is False
    )
    assert (
        file_ops.is_duplicate_operation("delete", "path/to/file3.txt", config) is True
    )


# Test logging a file operation
def test_log_operation(config: Config):
    file_ops.log_operation("log_test", "path/to/test", config)
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert f"log_test: path/to/test\n" in content


def test_text_checksum(file_content: str):
    checksum = file_ops.text_checksum(file_content)
    different_checksum = file_ops.text_checksum("other content")
    assert re.match(r"^[a-fA-F0-9]+$", checksum) is not None
    assert checksum != different_checksum


def test_log_operation_with_checksum(config: Config):
    file_ops.log_operation("log_test", "path/to/test", config, checksum="ABCDEF")
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert f"log_test: path/to/test #ABCDEF\n" in content


@pytest.mark.parametrize(
    "max_length, overlap, content, expected",
    [
        (
            4,
            1,
            "abcdefghij",
            ["abcd", "defg", "ghij"],
        ),
        (
            4,
            0,
            "abcdefghijkl",
            ["abcd", "efgh", "ijkl"],
        ),
        (
            4,
            0,
            "abcdefghijklm",
            ["abcd", "efgh", "ijkl", "m"],
        ),
        (
            4,
            0,
            "abcdefghijk",
            ["abcd", "efgh", "ijk"],
        ),
    ],
)
# Test splitting a file into chunks
def test_split_file(max_length, overlap, content, expected):
    assert (
        list(file_ops.split_file(content, max_length=max_length, overlap=overlap))
        == expected
    )


def test_read_file(
    mock_MemoryItem_from_text,
    test_file_with_content_path: Path,
    file_content,
    config: Config,
):
    content = file_ops.read_file(test_file_with_content_path, config)
    assert content.replace("\r", "") == file_content


def test_write_to_file(test_file_path: Path, config):
    new_content = "This is new content.\n"
    file_ops.write_to_file(str(test_file_path), new_content, config)
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


def test_write_file_logs_checksum(test_file_path: Path, config):
    new_content = "This is new content.\n"
    new_checksum = file_ops.text_checksum(new_content)
    file_ops.write_to_file(str(test_file_path), new_content, config)
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        log_entry = f.read()
    assert log_entry == f"write: {test_file_path} #{new_checksum}\n"


def test_write_file_fails_if_content_exists(test_file_path: Path, config):
    new_content = "This is new content.\n"
    file_ops.log_operation(
        "write",
        str(test_file_path),
        config,
        checksum=file_ops.text_checksum(new_content),
    )
    result = file_ops.write_to_file(str(test_file_path), new_content, config)
    assert result == "Error: File has already been updated."


def test_write_file_succeeds_if_content_different(
    test_file_with_content_path: Path, config
):
    new_content = "This is different content.\n"
    result = file_ops.write_to_file(
        str(test_file_with_content_path), new_content, config
    )
    assert result == "File written to successfully."


def test_append_to_file(test_nested_file: Path, config):
    append_text = "This is appended text.\n"
    file_ops.write_to_file(test_nested_file, append_text, config)

    file_ops.append_to_file(test_nested_file, append_text, config)

    with open(test_nested_file, "r") as f:
        content_after = f.read()

    assert content_after == append_text + append_text


def test_append_to_file_uses_checksum_from_appended_file(test_file_path: Path, config):
    append_text = "This is appended text.\n"
    file_ops.append_to_file(test_file_path, append_text, config)
    file_ops.append_to_file(test_file_path, append_text, config)
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        log_contents = f.read()

    digest = hashlib.md5()
    digest.update(append_text.encode("utf-8"))
    checksum1 = digest.hexdigest()
    digest.update(append_text.encode("utf-8"))
    checksum2 = digest.hexdigest()
    assert log_contents == (
        f"append: {test_file_path} #{checksum1}\n"
        f"append: {test_file_path} #{checksum2}\n"
    )


def test_delete_file(test_file_with_content_path: Path, config):
    result = file_ops.delete_file(str(test_file_with_content_path), config)
    assert result == "File deleted successfully."
    assert os.path.exists(test_file_with_content_path) is False


def test_delete_missing_file(config):
    filename = "path/to/file/which/does/not/exist"
    # confuse the log
    file_ops.log_operation("write", filename, config, checksum="fake")
    try:
        os.remove(filename)
    except FileNotFoundError as err:
        assert str(err) in file_ops.delete_file(filename, config)
        return
    assert False, f"Failed to test delete_file; {filename} not expected to exist"


def test_list_files(workspace: Workspace, test_directory: Path, config):
    # Case 1: Create files A and B, search for A, and ensure we don't return A and B
    file_a = workspace.get_path("file_a.txt")
    file_b = workspace.get_path("file_b.txt")

    with open(file_a, "w") as f:
        f.write("This is file A.")

    with open(file_b, "w") as f:
        f.write("This is file B.")

    # Create a subdirectory and place a copy of file_a in it
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    with open(os.path.join(test_directory, file_a.name), "w") as f:
        f.write("This is file A in the subdirectory.")

    files = file_ops.list_files(str(workspace.root), config)
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
    files = file_ops.list_files("", config)
    assert non_existent_file not in files


def test_download_file(workspace: Workspace, config):
    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.2.2.tar.gz"
    local_name = workspace.get_path("auto-gpt.tar.gz")
    size = 365023
    readable_size = readable_file_size(size)
    assert (
        file_ops.download_file(url, local_name, config)
        == f'Successfully downloaded and locally stored file: "{local_name}"! (Size: {readable_size})'
    )
    assert os.path.isfile(local_name) is True
    assert os.path.getsize(local_name) == size

    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.0.0.tar.gz"
    assert "Got an HTTP Error whilst trying to download file" in file_ops.download_file(
        url, local_name, config
    )

    url = "https://thiswebsiteiswrong.hmm/v0.0.0.tar.gz"
    assert "Failed to establish a new connection:" in file_ops.download_file(
        url, local_name, config
    )
