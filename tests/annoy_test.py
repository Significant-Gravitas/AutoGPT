import os
import sys
import unittest

from autogpt.memory.annoy import AnnoyMemory


def MockConfig():
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "memory_index": "auto-gpt",
            "annoy_index_file": "./annoy_index/annoy_index_file.ann",
            "annoy_metadata_file": "./annoy_index/annoy_metadata_file.json",
        },
    )


class TestAnnoyMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.memory = AnnoyMemory(self.cfg)

    def test_add(self):
        text = "Sample text"
        self.memory.add(text)
        self.assertIn(text, self.memory.metadata["texts"])

    def test_clear(self):
        self.memory.clear()
        self.assertEqual(self.memory.metadata, {"texts": [], "vectors": []})

    def test_get(self):
        text = "Sample text"
        self.memory.add(text)
        result = self.memory.get(text)
        self.assertEqual(result, [text])

    def test_get_relevant(self):
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.memory.add(text1)
        self.memory.add(text2)
        result = self.memory.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    def test_get_stats(self):
        text = "Sample text"
        self.memory.add(text)
        stats = self.memory.get_stats()
        self.assertEqual(stats, (1, self.memory.index.get_n_items()))


if __name__ == "__main__":
    unittest.main()
