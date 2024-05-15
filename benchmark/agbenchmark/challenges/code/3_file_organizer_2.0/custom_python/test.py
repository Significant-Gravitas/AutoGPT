import os
import subprocess
import tempfile
import unittest
import shutil


class TestOrganizeFiles(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Test files and their expected directory based on the first letter of their names
        self.test_files = [
            "apple.txt",
            "banana.txt",
            "Avocado.txt",
            "berry.png",
            "cherry.mp3",
        ]

        # Create test files
        for file_name in self.test_files:
            open(os.path.join(self.test_dir, file_name), "a").close()

    def test_organize_files_by_name(self):
        # Call the organize_files_by_name.py script using subprocess
        subprocess.call(
            ["python", "organize_files_by_name.py", "--directory_path=" + self.test_dir]
        )

        # Check if the files have been moved to the correct directories
        for file_name in self.test_files:
            first_letter = file_name[0].upper()
            self.assertTrue(
                os.path.isfile(os.path.join(self.test_dir, first_letter, file_name))
            )

    def tearDown(self):
        # Delete test directory and its contents
        for file_name in self.test_files:
            first_letter = file_name[0].upper()
            folder_path = os.path.join(self.test_dir, first_letter)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        if os.path.isdir(self.test_dir):
            os.rmdir(self.test_dir)


if __name__ == "__main__":
    unittest.main()
