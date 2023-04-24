""" Milvus memory storage provider."""
import re

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from autogpt.config import Config
from autogpt.llm_utils import get_ada_embedding
from autogpt.memory.base import MemoryProviderSingleton


class MilvusMemory(MemoryProviderSingleton):
    """Milvus memory storage provider."""

    def __init__(self, cfg: Config) -> None:
        """Construct a milvus memory storage connection.

        Args:
            cfg (Config): Auto-GPT global config.
        """
        self.configure(cfg)

        connect_kwargs = {}
        if self.username:
            connect_kwargs["user"] = self.username
            connect_kwargs["password"] = self.password

        connections.connect(
            **connect_kwargs,
            uri=self.uri or "",
            address=self.address or "",
            secure=self.secure,
        )

        self.init_collection()

    def configure(self, cfg: Config) -> None:
        # init with configuration.
        self.uri = None
        self.address = cfg.milvus_addr
        self.secure = cfg.milvus_secure
        self.username = cfg.milvus_username
        self.password = cfg.milvus_password
        self.collection_name = cfg.milvus_collection
        # use HNSW by default.
        self.index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        if (self.username is None) != (self.password is None):
            raise ValueError(
                "Both username and password must be set to use authentication for Milvus"
            )

        # configured address may be a full URL.
        if re.match(r"^(https?|tcp)://", self.address) is not None:
            self.uri = self.address
            self.address = None

            if self.uri.startswith("https"):
                self.secure = True

        # Zilliz Cloud requires AutoIndex.
        if re.match(r"^https://(.*)\.zillizcloud\.(com|cn)", self.address) is not None:
            self.index_params = {
                "metric_type": "IP",
                "index_type": "AUTOINDEX",
                "params": {},
            }

    def init_collection(self) -> None:
        """Initialize collection in vector database."""
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=65535),
        ]

        # create collection if not exist and load it.
        self.schema = CollectionSchema(fields, "auto-gpt memory storage")
        self.collection = Collection(self.collection_name, self.schema)
        # create index if not exist.
        if not self.collection.has_index():
            self.collection.release()
            self.collection.create_index(
                "embeddings",
                self.index_params,
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
        self.collection = Collection(self.collection_name, self.schema)
        self.collection.create_index(
            "embeddings",
            self.index_params,
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
