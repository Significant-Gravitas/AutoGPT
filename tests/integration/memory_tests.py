import unittest
import random
import string
import sys
from pathlib import Path
# Add the parent directory of the 'scripts' folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
from config import Config
from memory import LocalCache, RedisMemory

class TestLocalCache(unittest.TestCase):

    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    def setUp(self):
        cfg = Config()
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


class TestLocalCacheNamespace(unittest.TestCase):

    def setUp(self):
        cfg = Config()
        self.cache = LocalCache(cfg)
        self.cache.clear()

        self.ns1 = 'ns1'
        self.ns2 = 'ns2'

        # add test strings to the cache
        self.string_base = 'The quick brown fox jumps over the lazy '
        self.ns1_strings = [
            self.string_base + 'dog',
        ]
        self.ns2_strings = [
            self.string_base + 'cat',
        ]

        for text in self.ns1_strings:
            self.cache.add(text, namespace=self.ns1)

        for text in self.ns2_strings:
            self.cache.add(text, namespace=self.ns2)

    def test_get_relevant(self):
        d1 = self.cache.get_relevant(self.string_base, 2, namespace=self.ns1)
        d2 = self.cache.get_relevant(self.string_base, 2, namespace=self.ns2)

        self.assertEqual(len(d1), 1)
        self.assertEqual(len(d2), 1)

        self.assertIn(self.ns1_strings[0], d1)
        self.assertIn(self.ns2_strings[0], d2)


class TestRedisMemory(unittest.TestCase):
    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    def setUp(self):
        cfg = Config()
        cfg.wipe_redis_on_start = True
        self.cache = RedisMemory(cfg)

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


class TestRedisMemoryNamespace(unittest.TestCase):

    def setUp(self):
        cfg = Config()
        cfg.wipe_redis_on_start = True
        self.cache = RedisMemory(cfg)

        self.ns1 = 'ns1'
        self.ns2 = 'ns2'

        # add test strings to the cache
        self.string_base = 'The quick brown fox jumps over the lazy '
        self.ns1_strings = [
            self.string_base + 'dog',
        ]
        self.ns2_strings = [
            self.string_base + 'cat',
        ]

        for text in self.ns1_strings:
            self.cache.add(text, namespace=self.ns1)

        for text in self.ns2_strings:
            self.cache.add(text, namespace=self.ns2)

    def test_get_relevant(self):
        d1 = self.cache.get_relevant(self.string_base, 2, namespace=self.ns1)
        d2 = self.cache.get_relevant(self.string_base, 2, namespace=self.ns2)

        self.assertEqual(len(d1), 1)
        self.assertEqual(len(d2), 1)

        self.assertIn(self.ns1_strings[0], d1)
        self.assertIn(self.ns2_strings[0], d2)


if __name__ == '__main__':
    unittest.main()
