"""Tests the Vectara memory backend.

In order for running these tests, you need to set up the Vectara environment variables.
See https://github.com/Torantulino/Auto-GPT#-vectara-setup for details.
"""

import unittest

from autogpt.config import Config
from autogpt.memory.vectara import VectaraMemory


class TestVectaraMemory(unittest.TestCase):
    """Test the Vectara memory backend."""

    cfg = None
    index = None

    @classmethod
    def setUpClass(cls):
        """Set up the test environment for the VectaraMemory tests."""
        cls.cfg = Config()
        # Override the memory index to avoid accidentally polluting or clearing
        # actual index data.
        cls.cfg.memory_index = "Auto-GPT-Vectara-Unittest-Xq61LdhxQhupS146Ee43"

    def setUp(self):
        """Set up the memory backend."""
        self.memory = VectaraMemory(self.cfg)
        self.memory.clear()

    def test_get(self):
        """Test getting text from the memory backend."""
        self.memory.add("Cricket is a long but interesting game.")
        self.memory.add("The new avatar movie is out.")
        self.memory.add("Hiking keeps one healthy.")

        actual = self.memory.get("films")
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0], "The new avatar movie is out.")

    def test_clear(self):
        """Test clearing the memory backend."""
        self.memory.add("Hiking keeps one healthy.")
        actual = self.memory.get("films")
        self.assertEqual(len(actual), 1)

        self.memory.clear()
        actual = self.memory.get("films")
        self.assertEqual(len(actual), 0)


if __name__ == "__main__":
    unittest.main()
