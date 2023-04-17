import os
import sys
import unittest
from unittest import mock
from uuid import uuid4

from weaviate import Client
from weaviate.util import get_valid_uuid

from autogpt.config import Config
from autogpt.memory.base import get_ada_embedding
from autogpt.memory.weaviate import WeaviateMemory


class TestWeaviateMemory(unittest.TestCase):
    cfg = None
    client = None
    index = None

    @classmethod
    def setUpClass(cls):
        # only create the connection to weaviate once
        cls.cfg = Config()

        if cls.cfg.use_weaviate_embedded:
            from weaviate.embedded import EmbeddedOptions

            cls.client = Client(
                embedded_options=EmbeddedOptions(
                    hostname=cls.cfg.weaviate_host,
                    port=int(cls.cfg.weaviate_port),
                    persistence_data_path=cls.cfg.weaviate_embedded_path,
                )
            )
        else:
            cls.client = Client(
                f"{cls.cfg.weaviate_protocol}://{cls.cfg.weaviate_host}:{self.cfg.weaviate_port}"
            )

        cls.index = WeaviateMemory.format_classname(cls.cfg.memory_index)

    """
    In order to run these tests you will need a local instance of
    Weaviate running. Refer to https://weaviate.io/developers/weaviate/installation/docker-compose
    for creating local instances using docker.
    Alternatively in your .env file set the following environmental variables to run Weaviate embedded (see: https://weaviate.io/developers/weaviate/installation/embedded):

        USE_WEAVIATE_EMBEDDED=True
        WEAVIATE_EMBEDDED_PATH="/home/me/.local/share/weaviate"
    """

    def setUp(self):
        try:
            self.client.schema.delete_class(self.index)
        except:
            pass

        self.memory = WeaviateMemory(self.cfg)

    def test_add(self):
        doc = "You are a Titan name Thanos and you are looking for the Infinity Stones"
        self.memory.add(doc)
        result = self.client.query.get(self.index, ["raw_text"]).do()
        actual = result["data"]["Get"][self.index]

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["raw_text"], doc)

    def test_get(self):
        doc = "You are an Avenger and swore to defend the Galaxy from a menace called Thanos"

        with self.client.batch as batch:
            batch.add_data_object(
                uuid=get_valid_uuid(uuid4()),
                data_object={"raw_text": doc},
                class_name=self.index,
                vector=get_ada_embedding(doc),
            )

            batch.flush()

        actual = self.memory.get(doc)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0], doc)

    def test_get_stats(self):
        docs = [
            "You are now about to count the number of docs in this index",
            "And then you about to find out if you can count correctly",
        ]

        [self.memory.add(doc) for doc in docs]

        stats = self.memory.get_stats()

        self.assertTrue(stats)
        self.assertTrue("count" in stats)
        self.assertEqual(stats["count"], 2)

    def test_clear(self):
        docs = [
            "Shame this is the last test for this class",
            "Testing is fun when someone else is doing it",
        ]

        [self.memory.add(doc) for doc in docs]

        self.assertEqual(self.memory.get_stats()["count"], 2)

        self.memory.clear()

        self.assertEqual(self.memory.get_stats()["count"], 0)


if __name__ == "__main__":
    unittest.main()
