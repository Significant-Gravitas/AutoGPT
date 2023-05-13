from .abstract import ContextMemoryProvider
from .json_file import JSONFileMemory
from .no_memory import NoMemory
from .pinecone import PineconeMemory
from .redis import RedisMemory

__all__ = [
    "ContextMemoryProvider",
    "JSONFileMemory",
    "NoMemory",
    "PineconeMemory",
    "RedisMemory",
]

# add backends requiring libraries that are not installed by default
try:
    from .milvus import MilvusMemory

    __all__.append("MilvusMemory")
except ImportError:
    pass

try:
    from .weaviate import WeaviateMemory

    __all__.append("WeaviateMemory")
except ImportError:
    pass
