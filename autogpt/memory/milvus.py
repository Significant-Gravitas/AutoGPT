""" Milvus memory storage provider."""
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from regex import regex

from autogpt.memory.base import MemoryProviderSingleton, get_ada_embedding


class MilvusMemory(MemoryProviderSingleton):
    """Milvus memory storage provider."""

    def __init__(self, cfg) -> None:
        """Construct a milvus memory storage connection.

        Args:
            cfg (Config): Auto-GPT global config.
        """
        # parse from configuration.
        self.parse_configure(cfg)

        # use remote uri or addr to connect.
        if self.uri is not None:
            connections.connect(
                uri=self.uri,
                user=self.username,
                password=self.password,
                secure=self.secure,
            )
        else:
            connections.connect(
                address=self.address,
                user=self.username,
                password=self.password,
                secure=self.secure,
            )
        # init collection.
        self.init_collection()

    def parse_configure(self, cfg) -> None:
        # init with configuration.
        self.uri = None
        self.address = cfg.milvus_addr
        self.username = cfg.milvus_username
        self.password = cfg.milvus_password
        self.collection_name = cfg.milvus_collection
        self.secure = cfg.milvus_secure
        # use HNSW by default.
        self.index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        # check if config by url.
        re_uri_prefix = regex.compile(r"^(https?|tcp)://")
        if re_uri_prefix.match(self.address) is not None:
            self.uri = self.address
            # secure option should be True if https is enabled.
            if self.uri.startswith("https"):
                self.secure = True

        # zilliz cloud support AutoIndex but not manually index.
        re_zilliz_cloud = regex.compile(r"^https://(.*)\.zillizcloud\.(com|cn)")
        if re_zilliz_cloud.match(self.address) is not None:
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
