from forge.config.config import Config

from .memory_item import MemoryItem, MemoryItemFactory, MemoryItemRelevance
from .providers.base import VectorMemoryProvider as VectorMemory
from .providers.json_file import JSONFileMemory
from .providers.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ["json_file", "no_memory"]

# try:
#     from .providers.redis import RedisMemory

#     supported_memory.append("redis")
# except ImportError:
#     RedisMemory = None

# try:
#     from .providers.pinecone import PineconeMemory

#     supported_memory.append("pinecone")
# except ImportError:
#     PineconeMemory = None

# try:
#     from .providers.weaviate import WeaviateMemory

#     supported_memory.append("weaviate")
# except ImportError:
#     WeaviateMemory = None

# try:
#     from .providers.milvus import MilvusMemory

#     supported_memory.append("milvus")
# except ImportError:
#     MilvusMemory = None


def get_memory(config: Config) -> VectorMemory:
    """
    Returns a memory object corresponding to the memory backend specified in the config.

    The type of memory object returned depends on the value of the `memory_backend`
    attribute in the configuration. E.g. if `memory_backend` is set to "pinecone", a
    `PineconeMemory` object is returned. If it is set to "redis", a `RedisMemory`
    object is returned.
    By default, a `JSONFileMemory` object is returned.

    Params:
        config: A configuration object that contains information about the memory
            backend to be used and other relevant parameters.

    Returns:
        VectorMemory: an instance of a memory object based on the configuration provided
    """
    memory = None

    match config.memory_backend:
        case "json_file":
            memory = JSONFileMemory(config)

        case "pinecone":
            raise NotImplementedError(
                "The Pinecone memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added "
                "back in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not PineconeMemory:
            #     logger.warning(
            #         "Error: Pinecone is not installed. Please install pinecone"
            #         " to use Pinecone as a memory backend."
            #     )
            # else:
            #     memory = PineconeMemory(config)
            #     if clear:
            #         memory.clear()

        case "redis":
            raise NotImplementedError(
                "The Redis memory backend has been rendered incompatible by work on "
                "the memory system, and has been removed temporarily."
            )
            # if not RedisMemory:
            #     logger.warning(
            #         "Error: Redis is not installed. Please install redis-py to"
            #         " use Redis as a memory backend."
            #     )
            # else:
            #     memory = RedisMemory(config)

        case "weaviate":
            raise NotImplementedError(
                "The Weaviate memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added "
                "back in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not WeaviateMemory:
            #     logger.warning(
            #         "Error: Weaviate is not installed. Please install weaviate-client"
            #         " to use Weaviate as a memory backend."
            #     )
            # else:
            #     memory = WeaviateMemory(config)

        case "milvus":
            raise NotImplementedError(
                "The Milvus memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added "
                "back in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not MilvusMemory:
            #     logger.warning(
            #         "Error: pymilvus sdk is not installed, but required "
            #         "to use Milvus or Zilliz as memory backend. "
            #         "Please install pymilvus."
            #     )
            # else:
            #     memory = MilvusMemory(config)

        case "no_memory":
            memory = NoMemory()

        case _:
            raise ValueError(
                f"Unknown memory backend '{config.memory_backend}'."
                " Please check your config."
            )

    if memory is None:
        memory = JSONFileMemory(config)

    return memory


def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "get_memory",
    "MemoryItem",
    "MemoryItemFactory",
    "MemoryItemRelevance",
    "JSONFileMemory",
    "NoMemory",
    "VectorMemory",
    # "RedisMemory",
    # "PineconeMemory",
    # "MilvusMemory",
    # "WeaviateMemory",
]
