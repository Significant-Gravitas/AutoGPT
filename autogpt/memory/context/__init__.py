from autogpt.config import Config
from autogpt.logs import logger

from .memory_item import MemoryItem, MemoryItemRelevance
from .providers import ContextMemoryProvider as ContextMemory
from .providers.json_file import JSONFileMemory
from .providers.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ["json_file", "no_memory"]

try:
    from .providers.redis import RedisMemory

    supported_memory.append("redis")
except ImportError:
    RedisMemory = None

try:
    from .providers.pinecone import PineconeMemory

    supported_memory.append("pinecone")
except ImportError:
    PineconeMemory = None

try:
    from .providers.weaviate import WeaviateMemory

    supported_memory.append("weaviate")
except ImportError:
    WeaviateMemory = None

try:
    from .providers.milvus import MilvusMemory

    supported_memory.append("milvus")
except ImportError:
    MilvusMemory = None


def get_memory(cfg: Config, init=False) -> ContextMemory:
    memory = None

    match cfg.memory_backend:
        case "json_file":
            memory = JSONFileMemory(cfg)

        case "pinecone":
            if not PineconeMemory:
                logger.warn(
                    "Error: Pinecone is not installed. Please install pinecone"
                    " to use Pinecone as a memory backend."
                )
            else:
                memory = PineconeMemory(cfg)
                if init:
                    memory.clear()

        case "redis":
            if not RedisMemory:
                logger.warn(
                    "Error: Redis is not installed. Please install redis-py to"
                    " use Redis as a memory backend."
                )
            else:
                memory = RedisMemory(cfg)

        case "weaviate":
            if not WeaviateMemory:
                logger.warn(
                    "Error: Weaviate is not installed. Please install weaviate-client to"
                    " use Weaviate as a memory backend."
                )
            else:
                memory = WeaviateMemory(cfg)

        case "milvus":
            if not MilvusMemory:
                logger.warn(
                    "Error: pymilvus sdk is not installed."
                    "Please install pymilvus to use Milvus or Zilliz Cloud as memory backend."
                )
            else:
                memory = MilvusMemory(cfg)

        case "no_memory":
            memory = NoMemory(cfg)

        case _:
            raise ValueError(
                f"Unknown memory backend '{cfg.memory_backend}'. Please check your config."
            )

    if memory is None:
        memory = JSONFileMemory(cfg)

    return memory


def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "get_memory",
    "MemoryItem",
    "MemoryItemRelevance",
    "ContextMemory",
    "JSONFileMemory",
    "NoMemory",
    "RedisMemory",
    "PineconeMemory",
    "MilvusMemory",
    "WeaviateMemory",
]
