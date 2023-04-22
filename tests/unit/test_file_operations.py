import os
import shutil
import unittest
from pathlib import Path

from autogpt.commands.file_operations import (
    LOG_FILE_PATH,
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
from autogpt.workspace import path_in_workspace


class TestFileOperations(unittest.TestCase):
    """
    This set of unit tests is designed to test the file operations that autoGPT has access to.
    """

    def setUp(self):
        self.test_file = "test_file.txt"
        self.test_file2 = "test_file2.txt"
        self.test_directory = "test_directory"
        self.file_content = "This is a test file.\n"
        self.file_logger_logs = "file_logger.txt"

        with open(path_in_workspace(self.test_file), "w") as f:
            f.write(self.file_content)

        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)

    def tearDown(self):
        if os.path.exists(path_in_workspace(self.test_file)):
            os.remove(path_in_workspace(self.test_file))

        if os.path.exists(self.test_directory):
            shutil.rmtree(self.test_directory)

    def test_check_duplicate_operation(self):
        log_operation("write", self.test_file)
        self.assertTrue(check_duplicate_operation("write", self.test_file))

    # Test logging a file operation
    def test_log_operation(self):
        if os.path.exists(self.file_logger_logs):
            os.remove(self.file_logger_logs)

        log_operation("log_test", self.test_file)
        with open(LOG_FILE_PATH, "r") as f:
            content = f.read()
        self.assertIn("log_test: test_file.txt", content)

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
        with open(path_in_workspace(self.test_file), "r") as f:
            content = f.read()
        self.assertEqual(content, new_content)

    def test_append_to_file(self):
        with open(path_in_workspace(self.test_file), "r") as f:
            content_before = f.read()

        append_text = "This is appended text.\n"
        append_to_file(self.test_file, append_text)
        with open(path_in_workspace(self.test_file), "r") as f:
            content = f.read()

        self.assertEqual(content, content_before + append_text)

    def test_delete_file(self):
        delete_file(self.test_file)
        self.assertFalse(os.path.exists(path_in_workspace(self.test_file)))

    def test_search_files(self):
        # Case 1: Create files A and B, search for A, and ensure we don't return A and B
        file_a = "file_a.txt"
        file_b = "file_b.txt"

        with open(path_in_workspace(file_a), "w") as f:
            f.write("This is file A.")

        with open(path_in_workspace(file_b), "w") as f:
            f.write("This is file B.")

        # Create a subdirectory and place a copy of file_a in it
        if not os.path.exists(path_in_workspace(self.test_directory)):
            os.makedirs(path_in_workspace(self.test_directory))

        with open(
            path_in_workspace(os.path.join(self.test_directory, file_a)), "w"
        ) as f:
            f.write("This is file A in the subdirectory.")

        files = search_files(path_in_workspace(""))
        self.assertIn(file_a, files)
        self.assertIn(file_b, files)
        self.assertIn(os.path.join(self.test_directory, file_a), files)

        # Clean up
        os.remove(path_in_workspace(file_a))
        os.remove(path_in_workspace(file_b))
        os.remove(path_in_workspace(os.path.join(self.test_directory, file_a)))
        os.rmdir(path_in_workspace(self.test_directory))

        # Case 2: Search for a file that does not exist and make sure we don't throw
        non_existent_file = "non_existent_file.txt"
        files = search_files("")
        self.assertNotIn(non_existent_file, files)

    # Test to ensure we cannot read files out of workspace
    def test_restrict_workspace(self):
        CFG = Config()
        with open(self.test_file2, "w+") as f:
            f.write("test text")

        CFG.restrict_to_workspace = True

        # Get the absolute path of self.test_file2
        test_file2_abs_path = os.path.abspath(self.test_file2)

        with self.assertRaises(ValueError):
            read_file(test_file2_abs_path)

        CFG.restrict_to_workspace = False
        read_file(test_file2_abs_path)

        os.remove(test_file2_abs_path)


if __name__ == "__main__":
    unittest.main()
