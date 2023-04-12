import unittest
from unittest import mock
import sys
import os

from weaviate import Client
from weaviate.util import get_valid_uuid
from uuid import uuid4

sys.path.append(os.path.abspath('./scripts'))
from config import Config
from memory.weaviate import WeaviateMemory
from memory.base import get_ada_embedding

@mock.patch.dict(os.environ, {
    "WEAVIATE_HOST": "127.0.0.1",
    "WEAVIATE_PROTOCOL": "http",
    "WEAVIATE_PORT": "8080",
    "WEAVIATE_USERNAME": "",
    "WEAVIATE_PASSWORD": "",
    "MEMORY_INDEX": "AutogptTests"
})
class TestWeaviateMemory(unittest.TestCase):
    """
    In order to run these tests you will need a local instance of
    Weaviate running. Refer to https://weaviate.io/developers/weaviate/installation/docker-compose
    for creating local instances using docker.
    Alternatively in your .env file set the following environmental variables to run Weaviate embedded (see: https://weaviate.io/developers/weaviate/installation/embedded):
    
        USE_WEAVIATE_EMBEDDED=True
        WEAVIATE_EMBEDDED_PATH="/home/me/.local/share/weaviate"
    """
    def setUp(self):
        self.cfg = Config()

        if self.cfg.use_weaviate_embedded:
            from weaviate.embedded import EmbeddedOptions

            self.client = Client(embedded_options=EmbeddedOptions(
                hostname=self.cfg.weaviate_host,
                port=int(self.cfg.weaviate_port),
                persistence_data_path=self.cfg.weaviate_embedded_path
            ))
        else:
            self.client = Client(f"{self.cfg.weaviate_protocol}://{self.cfg.weaviate_host}:{self.cfg.weaviate_port}")

        try:
            self.client.schema.delete_class(self.cfg.memory_index)
        except:
            pass
        
        self.memory = WeaviateMemory(self.cfg)

    def test_add(self):
        doc = 'You are a Titan name Thanos and you are looking for the Infinity Stones'
        self.memory.add(doc)
        result = self.client.query.get(self.cfg.memory_index, ['raw_text']).do()
        actual = result['data']['Get'][self.cfg.memory_index]

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['raw_text'], doc)

    def test_get(self):
        doc = 'You are an Avenger and swore to defend the Galaxy from a menace called Thanos'
        
        with self.client.batch as batch:
            batch.add_data_object(
                uuid=get_valid_uuid(uuid4()),
                data_object={'raw_text': doc},
                class_name=self.cfg.memory_index,
                vector=get_ada_embedding(doc)
            )

            batch.flush()

        actual = self.memory.get(doc)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0], doc)


    def test_get_stats(self):
        docs = [
            'You are now about to count the number of docs in this index',
            'And then you about to find out if you can count correctly'
        ]

        [self.memory.add(doc) for doc in docs]

        stats = self.memory.get_stats()

        self.assertTrue(stats)
        self.assertTrue('count' in stats)
        self.assertEqual(stats['count'], 2)


    def test_clear(self):
        docs = [
            'Shame this is the last test for this class',
            'Testing is fun when someone else is doing it'
        ]

        [self.memory.add(doc) for doc in docs]

        self.assertEqual(self.memory.get_stats()['count'], 2)

        self.memory.clear()

        self.assertEqual(self.memory.get_stats()['count'], 0)


if __name__ == '__main__':
    unittest.main()
