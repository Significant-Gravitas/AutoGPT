"""
This set of unit tests is designed to test the file operations that autoGPT has access to.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from tempfile import gettempdir, NamedTemporaryFile
import pytest
from unittest.mock import patch
from autogpt.commands import file_operations

from autogpt.commands.file_operations import (
    append_to_file,
    file_operations_state,
    is_duplicate_operation,
    delete_file,
    download_file,
    log_operation,
    operations_from_log,
    read_file,
    search_files,
    split_file,
    write_to_file,
)
from autogpt.config import Config
from autogpt.utils import readable_file_size


@contextmanager
def temp_log_file():
    """Test utility to make the log file a temp file and patch it in the CFG."""
    with NamedTemporaryFile() as log_file:
        with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
            yield log_file


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


@pytest.fixture()
def test_file(workspace, file_content):
    test_file = str(workspace.get_path("test_file.txt"))
    with open(test_file, "w") as f:
        f.write(file_content)
    return test_file


@pytest.fixture()
def test_directory(workspace):
    return str(workspace.get_path("test_directory"))


@pytest.fixture()
def test_nested_file(workspace):
    return str(workspace.get_path("nested/test_file.txt"))

def test_file_operations():
    log_file_content = (
        b"File Operation Logger\n"
        b"write: /path/to/file1.txt #checksum1\n"
        b"write: /path/to/file2.txt #checksum2\n"
        b"write: /path/to/file3.txt #checksum3\n"
        b"append: /path/to/file2.txt #checksum4\n"
        b"delete: /path/to/file3.txt\n"
    )
    with NamedTemporaryFile(delete=False) as log_file:
        log_file.write(log_file_content)
        log_file.close()
    expected = [
        ("write", "/path/to/file1.txt", "checksum1"),
        ("write", "/path/to/file2.txt", "checksum2"),
        ("write", "/path/to/file3.txt", "checksum3"),
        ("append", "/path/to/file2.txt", "checksum4"),
        ("delete", "/path/to/file3.txt", None),
    ]
    assert list(operations_from_log(log_file.name)) == expected


def test_file_operations_state():
    # Prepare a fake log file
    log_file_content = (
        b"File Operation Logger\n"
        b"write: /path/to/file1.txt #checksum1\n"
        b"write: /path/to/file2.txt #checksum2\n"
        b"write: /path/to/file3.txt #checksum3\n"
        b"append: /path/to/file2.txt #checksum4\n"
        b"delete: /path/to/file3.txt\n"
    )
    with NamedTemporaryFile(delete=False) as log_file:
        log_file.write(log_file_content)
        log_file.close()
    # Call the function and check the returned dictionary
    expected_state = {
        "/path/to/file1.txt": "checksum1",
        "/path/to/file2.txt": "checksum4",
    }
    assert file_operations_state(log_file.name) == expected_state


def test_is_duplicate_operation():
    # Prepare a fake state dictionary for the function to use
    state = {
        "/path/to/file1.txt": "checksum1",
        "/path/to/file2.txt": "checksum2",
    }
    with patch.object(file_operations, "file_operations_state", lambda _: state):
        # Test cases with write operations
        assert is_duplicate_operation("write", "/path/to/file1.txt", "checksum1") is True
        assert is_duplicate_operation("write", "/path/to/file1.txt", "checksum2") is False
        assert is_duplicate_operation("write", "/path/to/file3.txt", "checksum3") is False
        # Test cases with append operations
        assert is_duplicate_operation("append", "/path/to/file1.txt", "checksum1") is False
        # Test cases with delete operations
        assert is_duplicate_operation("delete", "/path/to/file1.txt") is False
        assert is_duplicate_operation("delete", "/path/to/file3.txt") is True


# Test logging a file operation
def test_log_operation():
    with temp_log_file() as log_file:
        log_operation("log_test", "/path/to/test")
        with open(log_file.name, "r", encoding="utf-8") as f:
            content = f.read()
        assert f"log_test: /path/to/test\n" in content


def test_log_operation_with_checksum():
    with temp_log_file() as log_file:
        log_operation("log_test", "/path/to/test", checksum="ABCDEF")
        with open(log_file.name, "r", encoding="utf-8") as f:
            content = f.read()
        assert f"log_test: /path/to/test #ABCDEF\n" in content


# Test splitting a file into chunks
def test_split_file():
    content = "abcdefghij"
    chunks = list(split_file(content, max_length=4, overlap=1))
    expected = ["abcd", "defg", "ghij"]
    assert chunks == expected


def test_read_file(test_file, file_content):
    content = read_file(test_file)
    assert content == file_content


def test_write_to_file():
    new_content = "This is new content.\n"
    with NamedTemporaryFile() as test_file:
        with temp_log_file():
            write_to_file(test_file.name, new_content)
            with open(test_file.name, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == new_content


def test_write_file_adds_the_correct_checksum():
    new_content = "This is new content.\n"
    with temp_log_file() as log_file:
        with NamedTemporaryFile() as test_file:
            test_filename = test_file.name
            write_to_file(test_file.name, new_content)
        with open(log_file.name, "r") as f:
            content = f.read()
        assert content == f"write: {test_filename} #7988e6e8f558f2955105163e09cb53fe\n"

def test_write_file_fails_if_content_exists():
    new_content = "This is new content.\n"
    with temp_log_file() as _:
        with NamedTemporaryFile() as test_file:
            test_filename = test_file.name
            log_operation(
                "write",
                test_filename,
                checksum="7988e6e8f558f2955105163e09cb53fe",
            )
            result = write_to_file(test_file.name, new_content)
    assert result == "Error: File has already been updated."

def test_write_file_succeeds_if_checksum_different():
    new_content = "This is different content.\n"
    with temp_log_file() as _:
        with NamedTemporaryFile() as test_file:
            test_filename = test_file.name
            log_operation(
                "write",
                test_filename,
                checksum="7988e6e8f558f2955105163e09cb53fe",
            )
            result = write_to_file(test_file.name, new_content)
    assert result == "File written to successfully."


def test_append_to_file(test_nested_file):
    with temp_log_file() as _:
        append_text = "This is appended text.\n"
        write_to_file(test_nested_file, append_text)

        append_to_file(test_nested_file, append_text)

        with open(test_nested_file, "r") as f:
            content_after = f.read()

        assert content_after == append_text + append_text


def test_append_to_file_uses_checksum_from_appended_file():
    append_text = "This is appended text.\n"
    with temp_log_file() as log_file:
        with NamedTemporaryFile() as test_file:
            test_filename = test_file.name
            append_to_file(test_filename, append_text)
            append_to_file(test_filename, append_text)
        with open(log_file.name, "r", encoding="utf-8") as f:
            log_contents = f.read()
    digest = hashlib.md5()
    digest.update(append_text.encode("utf-8"))
    checksum1 = digest.hexdigest()
    digest.update(append_text.encode("utf-8"))
    checksum2 = digest.hexdigest()
    assert log_contents == (
        f"append: {test_filename} #{checksum1}\n"
        f"append: {test_filename} #{checksum2}\n"
    )

def test_delete_file():
    with NamedTemporaryFile(delete=False) as file_to_delete:
        file_to_delete.write("hello".encode("utf-8"))

    with temp_log_file() as _:
        log_operation("write", file_to_delete.name, "checksum")
        result = delete_file(file_to_delete.name)
    assert result == "File deleted successfully."
    assert os.path.exists(file_to_delete.name) is False


def test_delete_missing_file():
    filename = "/path/to/file/which/does/not/exist"
    with temp_log_file() as _:
        # confuse the log
        log_operation("write", filename, checksum="fake")
        try:
            os.remove(filename)
        except FileNotFoundError as err:
            assert str(err) in delete_file(filename)
            return
        assert True, "Failed to test delete_file"


def test_search_files(config, workspace, test_directory):
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

    files = search_files(str(workspace.root))
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
    files = search_files("")
    assert non_existent_file not in files


def test_download_file():
    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.2.2.tar.gz"
    local_name = os.path.join(gettempdir(), "auto-gpt.tar.gz")
    size = 365023
    readable_size = readable_file_size(size)
    assert (
        download_file(url, local_name)
        == f'Successfully downloaded and locally stored file: "{local_name}"! (Size: {readable_size})'
    )
    assert os.path.isfile(local_name) is True
    assert os.path.getsize(local_name) == size

    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.0.0.tar.gz"
    assert "Got an HTTP Error whilst trying to download file" in download_file(
        url, local_name
    )

    url = "https://thiswebsiteiswrong.hmm/v0.0.0.tar.gz"
    assert "Failed to establish a new connection:" in download_file(url, local_name)
