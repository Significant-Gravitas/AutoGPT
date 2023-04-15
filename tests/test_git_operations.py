import unittest
from autogpt.commands.git_operations import clone_repository


class TestGitClone(unittest.TestCase):
    def test_clone_repository(self):
        result = clone_repository(
            "https://github.com/BillSchumacher/Auto-GPT.git", "auto-gpt-cloned"
        )
        self.assertEqual(
            result, 
            "Cloned https://github.com/BillSchumacher/Auto-GPT.git to auto-gpt-cloned")
