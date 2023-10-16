import unittest
from unittest.mock import patch, Mock
import os
import shutil  # For cleanup

# Assuming the clone_repo function is in a file named my_module.py
from forge.sdk.abilities.git import clone_repo, commit_changes

class TestCloneRepo(unittest.TestCase):
    
    # @patch("my_module.Repo.clone_from")
    @patch("os.getenv")
    def test_clone_repo(self, mock_getenv, mock_clone_from):
        # Mock the return value of os.getenv to simulate an existing GITHUB_TOKEN
        mock_getenv.return_value = "fake_token"
        
        # Define the mock values
        mock_repo_url = "https://github.com/fluxcd/flux2.git"
        mock_dest_path = "./test_repo"

        # Call the function
        result = clone_repo(None, "test_task", "test_step", mock_repo_url, mock_dest_path)
        
        # Assertions
        self.assertIn("Successfully cloned", result)
        
        # Cleanup: Remove the mocked directory if it was created
        if os.path.exists(mock_dest_path):
            shutil.rmtree(mock_dest_path)


    # @patch("my_module.Repo")
    def test_commit_changes(self, mock_repo):
        # Mocking the repo object
        instance = mock_repo.return_value
        instance.is_dirty.return_value = True
        instance.git.add.return_value = None
        instance.git.commit.return_value = None

        # Define the mock values
        mock_commit_message = "Test commit"
        mock_repo_path = "./mock_repo_path"

        # Call the function
        result = commit_changes(None, "test_task", "test_step", mock_commit_message, mock_repo_path)
        
        # Assertions
        self.assertIn("Successfully committed changes", result)
        instance.git.add.assert_called_with(A=True)
        instance.git.commit.assert_called_with(m=mock_commit_message)
if __name__ == "__main__":
    unittest.main()
