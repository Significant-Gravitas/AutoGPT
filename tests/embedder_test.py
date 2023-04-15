import os
import sys

from autogpt.config import Config
from autogpt.memory.base import get_embedding

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
