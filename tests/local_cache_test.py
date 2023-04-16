import os
import sys
import unittest

from autogpt.memory.local import LocalCache


def MockConfig():
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


class TestLocalCache(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.cache = LocalCache(self.cfg)

    def test_add(self):
        text = "Sample text"
        self.cache.add(text)
        self.assertIn(text, self.cache.data.texts)

    def test_clear(self):
        self.cache.clear()
        self.assertEqual(self.cache.data, [""])

    def test_get(self):
        text = "Sample text"
        self.cache.add(text)
        result = self.cache.get(text)
        self.assertEqual(result, [text])

    def test_get_relevant(self):
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.cache.add(text1)
        self.cache.add(text2)
        result = self.cache.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    def test_get_stats(self):
        text = "Sample text"
        self.cache.add(text)
        stats = self.cache.get_stats()
        self.assertEqual(stats, (1, self.cache.data.embeddings.shape))


if __name__ == "__main__":
    unittest.main()
