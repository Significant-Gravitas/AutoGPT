"""LlamaIndex memory storage provider."""
from llama_index.indices.vector_store.base import GPTVectorStoreIndex
from llama_index.data_structs.node_v2 import Node, DocumentRelationship
from llama_index.indices.registry import INDEX_STRUCT_TYPE_TO_INDEX_CLASS
from typing import List, cast
import json
import uuid

from autogpt.memory.base import MemoryProviderSingleton
from autogpt.memory.base import get_ada_embedding
from autogpt.config import Config


class LlamaIndexMemory(MemoryProviderSingleton):
    """Llama Index memory storage provider."""

    def __init__(self, cfg: Config) -> None:
        """Initialize memory module.

        Args:
            cfg (Config): Auto-GPT global config.

        """
        self._struct_type = cfg.llamaindex_struct_type

        index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[self._struct_type]
        if cfg.llamaindex_json_path:
            index = index_cls.load_from_disk(cfg.llamaindex_json_path)
        else:
            # initialize a blank index (empty list of documents)
            index = index_cls([])
        # TODO: enforce that we use a vector store index right now
        if not isinstance(index, GPTVectorStoreIndex):
            raise ValueError("Index must be a vector store index.")
        self._index = cast(GPTVectorStoreIndex, index)

        if cfg.llamaindex_query_kwargs_path:
            query_kwargs = json.load(open(cfg.llamaindex_query_kwargs_path, 'r'))
        else:
            query_kwargs = {}
            
        self._query_kwargs = query_kwargs

        self._node_ids = []

    def add(self, data: str) -> str:
        """Add a embedding of data into memory.

        Args:
            data (str): The raw text to construct embedding index.

        Returns:
            str: log.
        """
        # set node_doc_id = source doc id
        doc_id = str(uuid.uuid4())
        # NOTE: set a fixed ref_doc_id for now to get it working
        node = Node(
            data, 
            doc_id=doc_id,
            embedding=get_ada_embedding(data), 
            relationships={DocumentRelationship.SOURCE: doc_id}
        )
        node_id = node.get_doc_id()

        self._index.insert_nodes([node])
        self._node_ids.append(node_id)
        log_text = f"Inserting data into index:\n data: {data}"
        return log_text

    def get(self, data: str) -> List[str]:
        """Return the most relevant data in memory.
        Args:
            data: The data to compare to.
        """
        return self.get_relevant(data, num_relevant=1)

    def get_relevant(self, data: str, num_relevant: int = 5) -> List[str]:
        """Return the most relevant data in memory."""
        self._query_kwargs.update({"similarity_top_k": num_relevant})

        response = self._index.query(
            data, 
            mode="default", 
            response_mode="no_text",
            **self._query_kwargs,
        )
        return [s.source_text for s in response.source_nodes]

    def clear(self) -> str:
        """Drop the index in memory.

        Returns:
            str: log.
        """
        for node_id in self._node_ids:
            self._index.delete(node_id)
            self._index.docstore.delete_document(node_id, raise_error=False)
        self._node_ids = []

        return "Index cleared."
        

    def get_stats(self) -> str:
        """
        Returns: The stats of the index.

        """
        return (
            "Number of nodes "
            f"(if using in-memory index): {len(self._node_ids)}"
        )
