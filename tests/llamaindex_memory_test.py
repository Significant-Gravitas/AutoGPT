# sourcery skip: snake-case-functions
"""Tests for the LlamaIndexMemory class."""
import json
import os
import sys
import unittest
from typing import cast

try:
    from llama_index import GPTSimpleVectorIndex

    from autogpt.memory.llama_index import LlamaIndexMemory

    def mock_config() -> dict:
        """Mock the Config class."""
        return type(
            "MockConfig",
            (object,),
            {
                "llamaindex_struct_type": "simple_dict",
                "llamaindex_json_path": "index.json",
                "llamaindex_query_kwargs_path": "query_kwargs.json",
            },
        )

    class TestLlamaIndexMemory(unittest.TestCase):
        """Tests for the LlamaIndexMemory class."""

        def setUp(self) -> None:
            """Set up the test environment."""
            # Create a simple vector index
            self.cfg = mock_config()
            index = GPTSimpleVectorIndex([])
            query_kwargs = {"similarity_top_k": 1}
            index.save_to_disk(self.cfg.llamaindex_json_path)
            json.dump(query_kwargs, open(self.cfg.llamaindex_query_kwargs_path, "w"))

            self.memory = LlamaIndexMemory(self.cfg)

        def test_add(self) -> None:
            """Test adding a text to the cache."""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual([text], result)

        def test_clear(self) -> None:
            """Test clearing the cache."""
            self.memory.clear()
            index = cast(GPTSimpleVectorIndex, self.memory._index)
            self.assertEqual(len(index.docstore.docs), 0)
            self.assertEqual(len(index.index_struct.nodes_dict), 0)

        def test_get(self) -> None:
            """Test getting a text from the cache."""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            result = self.memory.get(text)
            self.assertEqual(result, [text])

        def test_get_relevant(self) -> None:
            """Test getting relevant texts from the cache."""
            text1 = "Sample text 1"
            text2 = "Sample text 2"
            self.memory.clear()
            self.memory.add(text1)
            self.memory.add(text2)
            result = self.memory.get_relevant(text1, 1)
            self.assertEqual(result, [text1])

        def test_get_stats(self) -> None:
            """Test getting the stats of the cache."""
            text = "Sample text"
            self.memory.clear()
            self.memory.add(text)
            stats = self.memory.get_stats()
            self.assertEqual(45, len(stats))

        def tearDown(self) -> None:
            """Tear down the test environment."""
            os.remove(self.cfg.llamaindex_json_path)
            os.remove(self.cfg.llamaindex_query_kwargs_path)

except ImportError:
    print("llama-index not installed, skipping tests")
