import os
import shutil
import unittest
from pathlib import Path

from autogpt.commands.file_operations import (
    append_to_file,
    check_duplicate_operation,
    delete_file,
    log_operation,
    read_file,
    search_files,
    split_file,
    write_to_file,
)
from autogpt.config import Config
from autogpt.workspace import Workspace


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
        self.file_content = "This is a test file.\n"
        self.file_logger_logs = "file_logger.txt"

        with open(self.test_file, "w") as f:
            f.write(self.file_content)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_path)

    def test_check_duplicate_operation(self):
        log_operation("write", self.test_file)
        self.assertTrue(check_duplicate_operation("write", self.test_file))

    # Test logging a file operation
    def test_log_operation(self):
        if os.path.exists(self.file_logger_logs):
            os.remove(self.file_logger_logs)

        log_operation("log_test", self.test_file)
        with open(self.config.file_logger_path, "r") as f:
            content = f.read()
        self.assertIn(f"log_test: {self.test_file}", content)

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
        write_to_file(self.test_file, new_content)
        with open(self.test_file, "r") as f:
            content = f.read()
        self.assertEqual(content, new_content)

    def test_append_to_file(self):
        with open(self.test_file, "r") as f:
            content_before = f.read()

        append_text = "This is appended text.\n"
        append_to_file(self.test_file, append_text)
        with open(self.test_file, "r") as f:
            content = f.read()

        self.assertEqual(content, content_before + append_text)

    def test_delete_file(self):
        delete_file(self.test_file)
        self.assertFalse(os.path.exists(self.test_file))

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
