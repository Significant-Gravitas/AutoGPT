import random
import string
import sys
import unittest
from pathlib import Path

from autogpt.config import Config
from autogpt.memory.local import LocalCache


class TestLocalCache(unittest.TestCase):
    """Unit tests for the LocalCache class in memory/local.py."""

    def random_string(self, length):
        """
        Generate a random string of the given length.
        Args:
            length (int): The length of the string to generate.
        """
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    def setUp(self):
        """Set up the test environment for the LocalCache tests."""
        cfg = cfg = Config()
        self.cache = LocalCache(cfg)
        self.cache.clear()

        # Add example texts to the cache
        self.example_texts = [
            "The quick brown fox jumps over the lazy dog",
            "I love machine learning and natural language processing",
            "The cake is a lie, but the pie is always true",
            "ChatGPT is an advanced AI model for conversation",
        ]
        # Add the example texts to the cache
        for text in self.example_texts:
            self.cache.add(text)

        for _ in range(5):
            self.cache.add(self.random_string(10))

    def test_get_relevant(self):
        """Test getting relevant texts from the cache."""
        query = "I'm interested in artificial intelligence and NLP"
        k = 3
        relevant_texts = self.cache.get_relevant(query, k)

        print(f"Top {k} relevant texts for the query '{query}':")
        for i, text in enumerate(relevant_texts, start=1):
            print(f"{i}. {text}")

        self.assertEqual(len(relevant_texts), k)
        self.assertIn(self.example_texts[1], relevant_texts)


if __name__ == "__main__":
    unittest.main()
