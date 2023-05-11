# sourcery skip: snake-case-functions
"""Tests for the AnalyticDBMemory class."""

import unittest
from unittest.mock import MagicMock, patch

MockStats = 0


def MockAnalyticDBFetchone(*args, **kwargs):
    return [MockStats]


def MockAnalyticDBFetchall(*args, **kwargs):
    return [
        [
            "Sample text 1",
            "",
            "mock_index_name",
        ]
    ]


def MockAnalyticDBConnect(*args, **kwargs):
    mock_conn = MagicMock(name="MockAnalyticDBConnection")
    mock_conn.cursor().__enter__().fetchall.side_effect = MockAnalyticDBFetchall
    mock_conn.cursor().__enter__().fetchone.side_effect = MockAnalyticDBFetchone
    return mock_conn


def MockGetAdaEmbedding(text: str) -> list[float]:
    index = hash(text) % 1536
    result = [0.0] * 1536
    result[index] = 1.0
    return result


try:
    from autogpt.memory.analyticdb import AnalyticDBMemory

    def mock_config() -> dict:
        """Mock the Config class"""
        return type(
            "MockConfig",
            (object,),
            {
                "debug_mode": False,
                "continuous_mode": False,
                "speak_mode": False,
                "pg_host": "127.0.0.1",
                "pg_port": 5432,
                "pg_relation": "autogpt",
                "pg_database": "test",
                "pg_user": "test",
                "pg_password": "test",
            },
        )

    class TestAnalyticDBMemory(unittest.TestCase):
        """Tests for the AnalyticDB Memory class."""

        @patch("psycopg2cffi.connect", new=MockAnalyticDBConnect)
        def setUp(self) -> None:
            """Set up the test environment"""
            self.cfg = mock_config()
            self.memory = AnalyticDBMemory(self.cfg)

        @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
        def test_add(self) -> None:
            """Test adding a text to the cache"""
            text = "Sample text 1"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual([text], result)

        def test_clear(self) -> None:
            """Test clearing the cache"""
            self.memory.clear()
            stats = self.memory.get_stats()
            self.assertEqual("Entities num: 0", stats)

        @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
        def test_get(self) -> None:
            """Test getting a text from the cache"""
            text = "Sample text 1"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual(result, [text])

        @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
        def test_get_relevant(self) -> None:
            """Test getting relevant texts from the cache"""
            text1 = "Sample text 1"
            text2 = "Sample text 2"
            self.memory.clear()
            self.memory.add(text1)
            self.memory.add(text2)
            result = self.memory.get_relevant(text1, 1)
            self.assertEqual(result, [text1])

        @patch("autogpt.memory.analyticdb.get_ada_embedding", new=MockGetAdaEmbedding)
        @patch("tests.analyticdb_memory_test.MockStats", 1)
        def test_get_stats(self) -> None:
            """Test getting the cache stats"""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            stats = self.memory.get_stats()
            self.assertEqual("Entities num: 1", stats)

except Exception as e:
    print("AnalyticDB not installed, skipping tests")
