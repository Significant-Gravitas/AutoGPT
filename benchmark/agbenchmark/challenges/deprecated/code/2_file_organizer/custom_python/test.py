import os
import subprocess
import tempfile
import unittest


class TestOrganizeFiles(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # File types and their corresponding directory
        self.file_types = {
            "test_image.png": "images",
            "test_doc.txt": "documents",
            "test_audio.mp3": "audio",
        }

        # Create test files
        for file_name in self.file_types.keys():
            open(os.path.join(self.test_dir, file_name), "a").close()

    def test_organize_files(self):
        # Call the organize_files.py script using subprocess
        subprocess.call(
            ["python", "organize_files.py", "--directory_path=" + self.test_dir]
        )

        # Check if the files have been moved to the correct directories
        for file_name, directory in self.file_types.items():
            self.assertTrue(
                os.path.isfile(os.path.join(self.test_dir, directory, file_name))
            )

    def tearDown(self):
        # Delete test directory and its contents
        for file_name, directory in self.file_types.items():
            os.remove(os.path.join(self.test_dir, directory, file_name))
        for directory in set(self.file_types.values()):
            os.rmdir(os.path.join(self.test_dir, directory))
        os.rmdir(self.test_dir)


if __name__ == "__main__":
    unittest.main()
