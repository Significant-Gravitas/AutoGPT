import unittest
import random
import string
import sys
from pathlib import Path
# Add the parent directory of the 'scripts' folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
from config import Config
from memory.local import LocalCache

class TestLocalCache(unittest.TestCase):

    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    def setUp(self):
        cfg = cfg = Config()
        self.cache = LocalCache(cfg)
        self.cache.clear()

        # Add example texts to the cache
        self.example_texts = [
            'The quick brown fox jumps over the lazy dog',
            'I love machine learning and natural language processing',
            'The cake is a lie, but the pie is always true',
            'ChatGPT is an advanced AI model for conversation'
        ]

        for text in self.example_texts:
            self.cache.add(text)

        # Add some random strings to test noise
        for _ in range(5):
            self.cache.add(self.random_string(10))

    def test_get_relevant(self):
        query = "I'm interested in artificial intelligence and NLP"
        k = 3
        relevant_texts = self.cache.get_relevant(query, k)

        print(f"Top {k} relevant texts for the query '{query}':")
        for i, text in enumerate(relevant_texts, start=1):
            print(f"{i}. {text}")

        self.assertEqual(len(relevant_texts), k)
        self.assertIn(self.example_texts[1], relevant_texts)


if __name__ == '__main__':
    unittest.main()
