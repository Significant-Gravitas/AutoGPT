# sourcery skip: snake-case-functions
"""Tests for the MilvusMemory class."""
import random
import string
import unittest

from autogpt.config import Config
from autogpt.memory.milvus import MilvusMemory

try:

    class TestMilvusMemory(unittest.TestCase):
        """Tests for the MilvusMemory class."""

        def random_string(self, length: int) -> str:
            """Generate a random string of the given length."""
            return "".join(random.choice(string.ascii_letters) for _ in range(length))

        def setUp(self) -> None:
            """Set up the test environment."""
            cfg = Config()
            cfg.milvus_addr = "localhost:19530"
            self.memory = MilvusMemory(cfg)
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

            print(f"Top {k} relevant texts for the query '{query}':")
            for i, text in enumerate(relevant_texts, start=1):
                print(f"{i}. {text}")

            self.assertEqual(len(relevant_texts), k)
            self.assertIn(self.example_texts[1], relevant_texts)

except:
    print(
        "Skipping tests/integration/milvus_memory_tests.py as Milvus is not installed."
    )
