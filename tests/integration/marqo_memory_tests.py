# sourcery skip: snake-case-functions
"""Tests for the MarqoMemory class."""
import random
import string
import unittest

from autogpt.config import Config
from autogpt.memory.marqo import MarqoMemory

try:

    class TestMarqoMemory(unittest.TestCase):
        """Tests for the MarqoMemory class."""

        def random_string(self, length: int) -> str:
            """Generate a random string of the given length."""
            return "".join(random.choice(string.ascii_letters) for _ in range(length))

        def setUp(self) -> None:
            """Set up the test environment."""
            cfg = Config()
            cfg.marqo_url = "http://localhost:8882"
            cfg.marqo_index_name = "autogpt-test"
            self.memory = MarqoMemory(cfg)
            self.memory.clear()

            # Add example texts to the cache
            self.example_texts = [
                "The quick brown fox jumps over the lazy dog",
                "I love machine learning and natural language processing",
                "The cake is a lie, but the pie is always true",
                "ChatGPT is an advanced AI model for conversation",
            ]

            for text in self.example_texts:
                self.memory.add(text)

            # Add some random strings to test noise
            for _ in range(5):
                self.memory.add(self.random_string(10))

        def test_get_relevant(self) -> None:
            """Test getting relevant texts from the cache."""
            query = "I'm interested in artificial intelligence and NLP"
            num_relevant = 3
            relevant_texts = self.memory.get_relevant(query, num_relevant)

            print(f"Top {num_relevant} relevant texts for the query '{query}':")
            for i, text in enumerate(relevant_texts, start=1):
                print(f"{i}. {text}")

            self.assertEqual(len(relevant_texts), num_relevant)
            self.assertIn(self.example_texts[1], relevant_texts)

        def test_add(self) -> None:
            """Test adding a text to the cache"""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual([text], result)

        def test_clear(self) -> None:
            """Test clearing the cache"""
            self.memory.clear()
            self.assertEqual(self.memory.get_stats()["numberOfDocuments"], 0)

        def test_get(self) -> None:
            """Test getting a text from the cache"""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual(result, [text])

        def test_get_relevant(self) -> None:
            """Test getting relevant texts from the cache"""
            text1 = "Sample text 1"
            text2 = "Sample text 2"
            self.memory.clear()
            self.memory.add(text1)
            self.memory.add(text2)
            result = self.memory.get_relevant(text1, 1)
            self.assertEqual(result, [text1])

        def test_get_stats(self) -> None:
            """Test getting the cache stats"""
            text = "Sample text"
            self.memory.clear()
            for _ in range(3):
                self.memory.add(text)
            num_docs = self.memory.get_stats()["numberOfDocuments"]
            self.assertEqual(3, num_docs)

except:
    print("Skipping tests/integration/marqo_memory_tests.py as Marqo is not installed.")
