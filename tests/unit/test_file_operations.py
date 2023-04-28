import hashlib
import os
import shutil
from tempfile import NamedTemporaryFile, TemporaryFile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch
from autogpt.commands import file_operations

from autogpt.commands.file_operations import (
    append_to_file,
    file_operations_state,
    is_duplicate_operation,
    delete_file,
    log_operation,
    operations_from_log,
    read_file,
    search_files,
    split_file,
    write_to_file,
)
from autogpt.config import Config
from autogpt.workspace import Workspace


class TestFileOperationDupeChecking(unittest.TestCase):
    def test_file_operations(self):
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
        try:
            self.assertListEqual(list(operations_from_log(log_file.name)), expected)
        finally:
            os.unlink(log_file.name)

    def test_file_operations_state(self):
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
        # with patch("builtins.open", mock_open(read_data=log_file_content)):
        # Call the function and check the returned dictionary
        expected_state = {
            "/path/to/file1.txt": "checksum1",
            "/path/to/file2.txt": "checksum4",
        }
        try:
            self.assertDictEqual(file_operations_state(log_file.name), expected_state)
        finally:
            os.unlink(log_file.name)

    def test_is_duplicate_operation(self):
        # Prepare a fake state dictionary for the function to use
        state: Dict[str, str] = {
            "/path/to/file1.txt": "checksum1",
            "/path/to/file2.txt": "checksum2",
        }
        with patch.object(file_operations, "file_operations_state", lambda _: state):
            # Test cases with write operations
            self.assertTrue(
                is_duplicate_operation("write", "/path/to/file1.txt", "checksum1")
            )
            self.assertFalse(
                is_duplicate_operation("write", "/path/to/file1.txt", "checksum2")
            )
            self.assertFalse(
                is_duplicate_operation("write", "/path/to/file3.txt", "checksum3")
            )
            # Test cases with append operations
            self.assertFalse(
                is_duplicate_operation("append", "/path/to/file1.txt", "checksum1")
            )
            # Test cases with delete operations
            self.assertFalse(is_duplicate_operation("delete", "/path/to/file1.txt"))
            self.assertTrue(is_duplicate_operation("delete", "/path/to/file3.txt"))


class TestFileOperations(unittest.TestCase):
    """
    This set of unit tests is designed to test the file operations that autoGPT has access to.
    """

    def setUp(self):
        self.config = Config()
        workspace_path = os.path.join(os.path.dirname(__file__), "workspace")
        self.workspace_path = Workspace.make_workspace(workspace_path)
        self.config.workspace_path = workspace_path
        self.config.file_logger_path = os.path.join(workspace_path, "file_logger.txt")
        self.workspace = Workspace(workspace_path, restrict_to_workspace=True)

        self.test_file = str(self.workspace.get_path("test_file.txt"))
        self.test_file2 = "test_file2.txt"
        self.test_directory = str(self.workspace.get_path("test_directory"))
        self.test_nested_file = str(self.workspace.get_path("nested/test_file.txt"))
        self.file_content = "This is a test file.\n"
        self.file_logger_logs = "file_logger.txt"

        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write(self.file_content)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_path)

    # Test logging a file operation
    def test_log_operation(self):
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                log_operation("log_test", "/path/to/test")
                with open(self.config.file_logger_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertIn(f"log_test: /path/to/test\n", content)

    def test_log_operation_with_checksum(self):
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                log_operation("log_test", "/path/to/test", checksum="ABCDEF")
                with open(self.config.file_logger_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertIn(f"log_test: /path/to/test #ABCDEF\n", content)

    # Test splitting a file into chunks
    def test_split_file(self):
        content = "abcdefghij"
        chunks = list(split_file(content, max_length=4, overlap=1))
        expected = ["abcd", "defg", "ghij"]
        self.assertEqual(chunks, expected)

    def test_read_file(self):
        content = read_file(self.test_file)
        self.assertEqual(content, self.file_content)

    def test_write_to_file(self):
        new_content = "This is new content.\n"
        with NamedTemporaryFile() as temp:
            with patch.object(file_operations.CFG, "file_logger_path", temp.name):
                write_to_file(self.test_nested_file, new_content)
                with open(self.test_nested_file, "r") as f:
                    content = f.read()
                self.assertEqual(content, new_content)

    def test_write_file_adds_the_correct_checksum(self):
        new_content = "This is new content.\n"
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                with NamedTemporaryFile() as test_file:
                    test_filename = test_file.name
                    write_to_file(test_file.name, new_content)
            with open(log_file.name, "r") as f:
                content = f.read()
            self.assertEqual(
                content, f"write: {test_filename} #7988e6e8f558f2955105163e09cb53fe\n"
            )

    def test_write_file_fails_if_content_exists(self):
        new_content = "This is new content.\n"
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                with NamedTemporaryFile() as test_file:
                    test_filename = test_file.name
                    log_operation(
                        "write",
                        test_filename,
                        checksum="7988e6e8f558f2955105163e09cb53fe",
                    )
                    result = write_to_file(test_file.name, new_content)
        self.assertEqual(result, "Error: File has already been updated.")

    def test_write_file_succeeds_if_checksum_different(self):
        new_content = "This is different content.\n"
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                with NamedTemporaryFile() as test_file:
                    test_filename = test_file.name
                    log_operation(
                        "write",
                        test_filename,
                        checksum="7988e6e8f558f2955105163e09cb53fe",
                    )
                    result = write_to_file(test_file.name, new_content)
        self.assertEqual(result, "File written to successfully.")

    def test_append_to_file(self):
        append_text = "This is appended text.\n"
        append_to_file(self.test_nested_file, append_text)
        with open(self.test_nested_file, "r") as f:
            content = f.read()

        append_to_file(self.test_nested_file, append_text)

        with open(self.test_nested_file, "r") as f:
            content_after = f.read()

        self.assertEqual(content_after, append_text + append_text)

    def test_append_to_file_uses_checksum_from_appended_file(self):
        append_text = "This is appended text.\n"
        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
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
        self.assertEqual(
            log_contents,
            (
                f"append: {test_filename} #{checksum1}\n"
                f"append: {test_filename} #{checksum2}\n"
            ),
        )

    def test_delete_file(self):
        with NamedTemporaryFile(delete=False) as file_to_delete:
            file_to_delete.write("hello".encode("utf-8"))

        with NamedTemporaryFile() as log_file:
            with patch.object(file_operations.CFG, "file_logger_path", log_file.name):
                log_operation("write", file_to_delete.name, "checksum")
                result = delete_file(file_to_delete.name)
        try:
            self.assertEqual(result, "File deleted successfully.")
            self.assertFalse(os.path.exists(file_to_delete.name))
        finally:
            if os.path.exists(file_to_delete.name):
                os.unlink(file_to_delete.name)

    def test_search_files(self):
        # Case 1: Create files A and B, search for A, and ensure we don't return A and B
        file_a = self.workspace.get_path("file_a.txt")
        file_b = self.workspace.get_path("file_b.txt")

        with open(file_a, "w") as f:
            f.write("This is file A.")

        with open(file_b, "w") as f:
            f.write("This is file B.")

        # Create a subdirectory and place a copy of file_a in it
        if not os.path.exists(self.test_directory):
            os.makedirs(self.test_directory)

        with open(os.path.join(self.test_directory, file_a.name), "w") as f:
            f.write("This is file A in the subdirectory.")

        files = search_files(str(self.workspace.root))
        self.assertIn(file_a.name, files)
        self.assertIn(file_b.name, files)
        self.assertIn(os.path.join(Path(self.test_directory).name, file_a.name), files)

        # Clean up
        os.remove(file_a)
        os.remove(file_b)
        os.remove(os.path.join(self.test_directory, file_a.name))
        os.rmdir(self.test_directory)

        # Case 2: Search for a file that does not exist and make sure we don't throw
        non_existent_file = "non_existent_file.txt"
        files = search_files("")
        self.assertNotIn(non_existent_file, files)


if __name__ == "__main__":
    unittest.main()
