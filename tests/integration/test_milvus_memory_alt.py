# sourcery skip: snake-case-functions
"""Tests for the MilvusMemory class."""
import random
import string

import pytest

pytest.importorskip("pymilvus", "2.2.0", "Pymilvus is not installed")

from autogpt.config import Config
from autogpt.memory.milvus import MilvusMemory


@pytest.fixture
def mock_config():
    cfg = Config()
    cfg.milvus_addr = "localhost:19530"
    return cfg


@pytest.mark.integration_test
class TestMilvusMemory:
    """Unit tests for the MilvusMemory class."""

    def generate_random_string(self, length: int) -> str:
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    @pytest.fixture(scope="class")
    def memory(self, mock_config: Config) -> MilvusMemory:
        memory = MilvusMemory(mock_config)
        memory.clear()

        # Add example texts to the cache
        self.example_texts = [
            "The quick brown fox jumps over the lazy dog",
            "I love machine learning and natural language processing",
            "The cake is a lie, but the pie is always true",
            "ChatGPT is an advanced AI model for conversation",
        ]

        for text in self.example_texts:
            memory.add(text)

        # Add some random strings to test noise
        for _ in range(5):
            memory.add(self.generate_random_string(10))

        return memory

    def test_get_relevant(self, memory: MilvusMemory) -> None:
        """Test getting relevant texts from the cache."""
        query = "I'm interested in artificial intelligence and NLP"
        num_relevant = 3
        relevant_texts = memory.get_relevant(query, num_relevant)

        print(f"Top {num_relevant} relevant texts for the query '{query}':")
        for i, text in enumerate(relevant_texts, start=1):
            print(f"{i}. {text}")

        assert len(relevant_texts) == num_relevant
        assert self.example_texts[1] in relevant_texts
