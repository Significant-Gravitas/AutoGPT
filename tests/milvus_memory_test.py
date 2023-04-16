import os
import sys
import unittest

from autogpt.memory.milvus import MilvusMemory


def MockConfig():
    return type(
        "MockConfig",
        (object,),
        {
            "debug_mode": False,
            "continuous_mode": False,
            "speak_mode": False,
            "milvus_collection": "autogpt",
            "milvus_addr": "localhost:19530",
        },
    )


class TestMilvusMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.memory = MilvusMemory(self.cfg)

    def test_add(self):
        text = "Sample text"
        self.memory.clear()
        self.memory.add(text)
        result = self.memory.get(text)
        self.assertEqual([text], result)

    def test_clear(self):
        self.memory.clear()
        self.assertEqual(self.memory.collection.num_entities, 0)

    def test_get(self):
        text = "Sample text"
        self.memory.clear()
        self.memory.add(text)
        result = self.memory.get(text)
        self.assertEqual(result, [text])

    def test_get_relevant(self):
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        self.memory.clear()
        self.memory.add(text1)
        self.memory.add(text2)
        result = self.memory.get_relevant(text1, 1)
        self.assertEqual(result, [text1])

    def test_get_stats(self):
        text = "Sample text"
        self.memory.clear()
        self.memory.add(text)
        stats = self.memory.get_stats()
        self.assertEqual(15, len(stats))


if __name__ == "__main__":
    unittest.main()
