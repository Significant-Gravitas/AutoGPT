"""
This module provides helper functions to manage memory backends, such as LocalCache,
RedisMemory, PineconeMemory, and NoMemory. It also lists the supported memory backends.
"""

from memory.local import LocalCache
from memory.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ['local']

try:
    from memory.redismem import RedisMemory
    supported_memory.append('redis')
except ImportError:
    print("Redis not installed. Skipping import.")
    RedisMemory = None

try:
    from memory.pinecone import PineconeMemory
    supported_memory.append('pinecone')
except ImportError:
    print("Pinecone not installed. Skipping import.")
    PineconeMemory = None


def get_memory(cfg):
    if cfg.memory_backend == "pinecone":
        if not PineconeMemory:
            print("Error: Pinecone is not installed. Please install pinecone"
                  " to use Pinecone as a memory backend.")
        else:
            return PineconeMemory(cfg)
    elif cfg.memory_backend == "redis":
        if not RedisMemory:
            print("Error: Redis is not installed. Please install redis-py to"
                  " use Redis as a memory backend.")
        else:
            return RedisMemory(cfg)
    elif cfg.memory_backend == "no_memory":
        return NoMemory(cfg)
    else:
        return LocalCache(cfg)


def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "get_memory",
    "LocalCache",
    "RedisMemory",
    "PineconeMemory",
    "NoMemory"
]
