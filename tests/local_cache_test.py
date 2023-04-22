# sourcery skip: snake-case-functions
"""Tests for LocalCache class"""
import os
import sys
import unittest

import pytest

from autogpt.memory.local import LocalCache
from tests.utils import requires_api_key


def mock_config() -> dict:
    """Mock the Config class"""
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "memory_index": "auto-gpt",
        },
    )


@pytest.mark.integration_test
class TestLocalCache(unittest.TestCase):
    """Tests for LocalCache class"""

    def setUp(self) -> None:
        """Set up the test environment"""
        self.cfg = mock_config()
        self.cache = LocalCache(self.cfg)

    @requires_api_key("OPENAI_API_KEY")
    def test_add(self) -> None:
        """Test adding a text to the cache"""
        text = "Sample text"
        self.cache.add(text)
        self.assertIn(text, self.cache.data.texts)

    @requires_api_key("OPENAI_API_KEY")
    def test_clear(self) -> None:
        """Test clearing the cache"""
        self.cache.clear()
        self.assertEqual(self.cache.data.texts, [])

    @requires_api_key("OPENAI_API_KEY")
    def test_get(self) -> None:
        """Test getting a text from the cache"""
        text = "Sample text"
        self.cache.add(text)
        result = self.cache.get(text)
        self.assertEqual(result, [text])

    @requires_api_key("OPENAI_API_KEY")
    def test_get_relevant(self) -> None:
        """Test getting relevant texts from the cache"""
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.cache.add(text1)
        self.cache.add(text2)
        result = self.cache.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    @requires_api_key("OPENAI_API_KEY")
    def test_get_stats(self) -> None:
        """Test getting the cache stats"""
        text = "Sample text"
        self.cache.add(text)
        stats = self.cache.get_stats()
        self.assertEqual(stats, (4, self.cache.data.embeddings.shape))
