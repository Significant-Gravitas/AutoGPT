import random
import string
import unittest
from unittest.mock import MagicMock, patch

from autogpt.config import Config
from autogpt.memory.analyticdb import AnalyticDBMemory


def MockAnalyticDBFetchall(*args, **kwargs):
    return [
        [
            "I love machine learning and natural language processing",
            "",
            "mock_index_name",
        ]
    ] + [["mock_result", "", "mock_index_name"]] * 2


def MockAnalyticDBConnect(*args, **kwargs):
    mock_conn = MagicMock(name="MockAnalyticDBConnection")
    mock_conn.cursor().__enter__().fetchall.side_effect = MockAnalyticDBFetchall
    return mock_conn


def MockGetAdaEmbedding(text: str) -> list[float]:
    index = hash(text) % 1536
    result = [0.0] * 1536
    result[index] = 1.0
    return result


class TestAnalyticDBMemory(unittest.TestCase):
    def random_string(self, length):
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    @patch("psycopg2cffi.connect", new=MockAnalyticDBConnect)
    @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
    def setUp(self):
        cfg = Config()
        cfg.pg_host = "your_analyticdb_host"
        cfg.pg_port = 5432
        cfg.pg_database = "test"
        cfg.pg_relation = "auto_gpt_test"
        cfg.pg_user = "test"
        cfg.pg_password = "test"
        self.memory = AnalyticDBMemory(cfg)
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

    @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
    def test_get_relevant(self):
        query = "I'm interested in artificial intelligence and NLP"
        k = 3
        relevant_texts = self.memory.get_relevant(query, k)

        print(f"Top {k} relevant texts for the query '{query}':")
        for i, text in enumerate(relevant_texts, start=1):
            print(f"{i}. {text}")

        self.assertEqual(len(relevant_texts), k)
        self.assertIn(self.example_texts[1], relevant_texts)


if __name__ == "__main__":
    unittest.main()
