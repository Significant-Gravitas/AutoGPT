""" Milvus memory storage provider."""
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from autogpt.memory.base import MemoryProviderSingleton, get_ada_embedding


class MilvusMemory(MemoryProviderSingleton):
    """Milvus memory storage provider."""

    def __init__(self, cfg) -> None:
        """Construct a milvus memory storage connection.

        Args:
            cfg (Config): Auto-GPT global config.
        """
        # connect to milvus server.
        connections.connect(address=cfg.milvus_addr)
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=65535),
        ]

        # create collection if not exist and load it.
        self.milvus_collection = cfg.milvus_collection
        self.schema = CollectionSchema(fields, "auto-gpt memory storage")
        self.collection = Collection(self.milvus_collection, self.schema)
        # create index if not exist.
        if not self.collection.has_index():
            self.collection.release()
            self.collection.create_index(
                "embeddings",
                {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                },
                index_name="embeddings",
            )
        self.collection.load()

    def add(self, data) -> str:
        """Add an embedding of data into memory.

        Args:
            data (str): The raw text to construct embedding index.

        Returns:
            str: log.
        """
        embedding = get_ada_embedding(data)
        result = self.collection.insert([[embedding], [data]])
        _text = (
            "Inserting data into memory at primary key: "
            f"{result.primary_keys[0]}:\n data: {data}"
        )
        return _text

    def get(self, data):
        """Return the most relevant data in memory.
        Args:
            data: The data to compare to.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """Drop the index in memory.

        Returns:
            str: log.
        """
        self.collection.drop()
        self.collection = Collection(self.milvus_collection, self.schema)
        self.collection.create_index(
            "embeddings",
            {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            },
            index_name="embeddings",
        )
        self.collection.load()
        return "Obliviated"

    def get_relevant(self, data: str, num_relevant: int = 5):
        """Return the top-k relevant data in memory.
        Args:
            data: The data to compare to.
            num_relevant (int, optional): The max number of relevant data.
                Defaults to 5.

        Returns:
            list: The top-k relevant data.
        """
        # search the embedding and return the most relevant text.
        embedding = get_ada_embedding(data)
        search_params = {
            "metrics_type": "IP",
            "params": {"nprobe": 8},
        }
        result = self.collection.search(
            [embedding],
            "embeddings",
            search_params,
            num_relevant,
            output_fields=["raw_text"],
        )
        return [item.entity.value_of_field("raw_text") for item in result[0]]

    def get_stats(self) -> str:
        """
        Returns: The stats of the milvus cache.
        """
        return f"Entities num: {self.collection.num_entities}"
