import os
import sys
# Probably a better way:
sys.path.append(os.path.abspath('../scripts'))
from memory.base import get_embedding
from config import Config
import unittest


# Required, because the get_embedding function uses it
cfg = Config()


class TestMemoryEmbedder(unittest.TestCase):
    def test_ada(self):
        cfg.memory_embedder = "ada"
        text = "Sample text"
        result = get_embedding(text)
        self.assertEqual(len(result), 1536)

    def test_sbert(self):
        cfg.memory_embedder = "sbert"
        text = "Sample text"
        result = get_embedding(text)
        self.assertEqual(len(result), 768)


if __name__ == '__main__':
    unittest.main()
