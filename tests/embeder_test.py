"""Tests for different embeders"""

import unittest
import os

from autogpt.memory.base import get_embedding


def mock_config() -> dict:
    """Mock the Config class"""
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "memory_embeder": "ada",
        },
    )


@unittest.skipIf(os.getenv("CI"), "Skipping memory embeder tests")
class TestEmbeder(unittest.TestCase):
    """Tests for different embeders"""

    def setUp(self) -> None:
        """Set up the test environment"""
        self.cfg = mock_config()

    def test_ada(self) -> None:
        """Test getting the embedding for a given text"""
        self.cfg.memory_embeder = "ada"
        text = "Sample text"
        embedding = get_embedding(text)
        self.assertEqual(len(embedding), 1536)

    def test_sbert(self) -> None:
        """Test getting the embedding for a given text using sBERT"""
        self.cfg.memory_embeder = "sbert"
        text = "Sample text"
        embedding = get_embedding(text)
        self.assertEqual(len(embedding), 768)
