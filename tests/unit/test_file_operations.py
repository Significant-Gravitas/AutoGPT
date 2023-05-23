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
from autogpt.utils import readable_file_size
from autogpt.workspace import Workspace


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


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
def test_file_with_content_path(test_file: TextIOWrapper, file_content):
    test_file.write(file_content)
    test_file.close()
    file_ops.log_operation(
        "write", test_file.name, file_ops.text_checksum(file_content)
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


def test_is_duplicate_operation(config, mocker: MockerFixture):
    # Prepare a fake state dictionary for the function to use
    state = {
        "path/to/file1.txt": "checksum1",
        "path/to/file2.txt": "checksum2",
    }
    mocker.patch.object(file_ops, "file_operations_state", lambda _: state)

    # Test cases with write operations
    assert (
        file_ops.is_duplicate_operation("write", "path/to/file1.txt", "checksum1")
        is True
    )
    assert (
        file_ops.is_duplicate_operation("write", "path/to/file1.txt", "checksum2")
        is False
    )
    assert (
        file_ops.is_duplicate_operation("write", "path/to/file3.txt", "checksum3")
        is False
    )
    # Test cases with append operations
    assert (
        file_ops.is_duplicate_operation("append", "path/to/file1.txt", "checksum1")
        is False
    )
    # Test cases with delete operations
    assert file_ops.is_duplicate_operation("delete", "path/to/file1.txt") is False
    assert file_ops.is_duplicate_operation("delete", "path/to/file3.txt") is True


# Test logging a file operation
def test_log_operation(config: Config):
    file_ops.log_operation("log_test", "path/to/test")
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert f"log_test: path/to/test\n" in content


def test_text_checksum(file_content: str):
    checksum = file_ops.text_checksum(file_content)
    different_checksum = file_ops.text_checksum("other content")
    assert re.match(r"^[a-fA-F0-9]+$", checksum) is not None
    assert checksum != different_checksum


def test_log_operation_with_checksum(config: Config):
    file_ops.log_operation("log_test", "path/to/test", checksum="ABCDEF")
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


def test_read_file(test_file_with_content_path: Path, file_content):
    content = file_ops.read_file(test_file_with_content_path)
    assert content == file_content


def test_write_to_file(test_file_path: Path):
    new_content = "This is new content.\n"
    file_ops.write_to_file(str(test_file_path), new_content)
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == new_content


def test_write_file_logs_checksum(config: Config, test_file_path: Path):
    new_content = "This is new content.\n"
    new_checksum = file_ops.text_checksum(new_content)
    file_ops.write_to_file(str(test_file_path), new_content)
    with open(config.file_logger_path, "r", encoding="utf-8") as f:
        log_entry = f.read()
    assert log_entry == f"write: {test_file_path} #{new_checksum}\n"


def test_write_file_fails_if_content_exists(test_file_path: Path):
    new_content = "This is new content.\n"
    file_ops.log_operation(
        "write",
        str(test_file_path),
        checksum=file_ops.text_checksum(new_content),
    )
    result = file_ops.write_to_file(str(test_file_path), new_content)
    assert result == "Error: File has already been updated."


def test_write_file_succeeds_if_content_different(test_file_with_content_path: Path):
    new_content = "This is different content.\n"
    result = file_ops.write_to_file(str(test_file_with_content_path), new_content)
    assert result == "File written to successfully."


def test_append_to_file(test_nested_file: Path):
    append_text = "This is appended text.\n"
    file_ops.write_to_file(test_nested_file, append_text)

    file_ops.append_to_file(test_nested_file, append_text)

    with open(test_nested_file, "r") as f:
        content_after = f.read()

    assert content_after == append_text + append_text


def test_append_to_file_uses_checksum_from_appended_file(
    config: Config, test_file_path: Path
):
    append_text = "This is appended text.\n"
    file_ops.append_to_file(test_file_path, append_text)
    file_ops.append_to_file(test_file_path, append_text)
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


def test_delete_file(test_file_with_content_path: Path):
    result = file_ops.delete_file(str(test_file_with_content_path))
    assert result == "File deleted successfully."
    assert os.path.exists(test_file_with_content_path) is False


def test_delete_missing_file(config):
    filename = "path/to/file/which/does/not/exist"
    # confuse the log
    file_ops.log_operation("write", filename, checksum="fake")
    try:
        os.remove(filename)
    except FileNotFoundError as err:
        assert str(err) in file_ops.delete_file(filename)
        return
    assert False, f"Failed to test delete_file; {filename} not expected to exist"


def test_list_files(workspace: Workspace, test_directory: Path):
    # Create different files with varying extensions, filenames and contents
    file_a = workspace.get_path("file_a.txt")
    file_b = workspace.get_path("file_b.pdf")
    file_c = workspace.get_path("sample_file_a.txt")
    file_d = workspace.get_path("file_D.txt")
    dot_file = workspace.get_path(".dotfile.txt")

    with open(file_a, "w") as f:
        f.write("This is file A.")
    with open(file_b, "w") as f:
        f.write("This is file B.")
    with open(file_c, "w") as f:
        f.write("This is another file A.")
    with open(file_d, "w") as f:
        f.write("This is file D.")
    with open(dot_file, "w") as f:
        f.write("This is a dotfile.")

    # Test extension filter with additional irrelevant filters (priority test)
    files = file_ops.list_files(
        str(workspace.root),
        extension=".txt",
        filename="file_b",
        query="This is file B.",
    )
    assert file_a.name in files
    assert file_c.name in files
    assert file_d.name in files
    assert file_b.name not in files

    # Test filename filter with additional irrelevant filters (priority test)
    files = file_ops.list_files(
        str(workspace.root), filename="file_a.txt", query="This is file B."
    )
    assert file_a.name in files
    assert file_c.name in files  # file_c should be included
    assert file_b.name not in files
    assert file_d.name not in files

    # Test case insensitivity
    files = file_ops.list_files(str(workspace.root), filename="FILE_D")
    assert file_d.name not in files

    # Test keywords filter with multiple keywords
    files = file_ops.list_files(
        str(workspace.root), keywords=["file_a", "file_b", "file_D"]
    )
    assert file_a.name in files
    assert file_b.name in files
    assert file_c.name in files
    assert file_d.name in files

    # Test query filter
    files = file_ops.list_files(str(workspace.root), query="file_a")
    assert file_a.name in files
    assert file_c.name in files
    assert file_b.name not in files
    assert file_d.name not in files

    # Test filename_query filter
    files = file_ops.list_files(str(workspace.root), filename_query="file_a")
    assert file_a.name in files
    assert file_c.name in files
    assert file_b.name not in files
    assert file_d.name not in files

    # Test filename_substring filter
    files = file_ops.list_files(str(workspace.root), filename_substring="file_a")
    assert file_a.name in files
    assert file_c.name in files
    assert file_b.name not in files
    assert file_d.name not in files

    # Test ignore files starting with "."
    files = file_ops.list_files(str(workspace.root))
    assert dot_file.name not in files

    # Cleanup
    os.remove(file_a)
    os.remove(file_b)
    os.remove(file_c)
    os.remove(file_d)
    os.remove(dot_file)

    # Test no filters
    files = file_ops.list_files(str(workspace.root))
    assert file_a.name not in files
    assert file_b.name not in files
    assert file_c.name not in files
    assert file_d.name not in files
    assert dot_file.name not in files

    # Search for a file that does not exist and make sure we don't throw an Exception
    non_existent_file = "non_existent_file.txt"
    files = file_ops.list_files(str(workspace.root))
    assert non_existent_file not in files


def test_download_file(config, workspace: Workspace):
    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.2.2.tar.gz"
    local_name = workspace.get_path("auto-gpt.tar.gz")
    size = 365023
    readable_size = readable_file_size(size)
    assert (
        file_ops.download_file(url, local_name)
        == f'Successfully downloaded and locally stored file: "{local_name}"! (Size: {readable_size})'
    )
    assert os.path.isfile(local_name) is True
    assert os.path.getsize(local_name) == size

    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.0.0.tar.gz"
    assert "Got an HTTP Error whilst trying to download file" in file_ops.download_file(
        url, local_name
    )

    url = "https://thiswebsiteiswrong.hmm/v0.0.0.tar.gz"
    assert "Failed to establish a new connection:" in file_ops.download_file(
        url, local_name
    )
