# sourcery skip: snake-case-functions
"""Tests for the AnalyticDBMemory class."""

import unittest

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

        def setUp(self) -> None:
            """Set up the test environment"""
            self.cfg = mock_config()
            self.memory = AnalyticDBMemory(self.cfg)

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
            stats = self.memory.get_stats()
            self.assertEqual("Entities num: 0", stats)

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
            self.memory.add(text)
            stats = self.memory.get_stats()
            self.assertEqual("Entities num: 1", stats)

except:
    print("AnalyticDB not installed, skipping tests")
