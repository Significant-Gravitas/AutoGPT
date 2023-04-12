from memory.local import LocalCache
from internal_print import print_to_console
try:
    from memory.redismem import RedisMemory
except ImportError:
    internal_print("Redis not installed. Skipping import.")
    RedisMemory = None

try:
    from memory.pinecone import PineconeMemory
except ImportError:
    internal_print("Pinecone not installed. Skipping import.")
    PineconeMemory = None


def get_memory(cfg, init=False):
    memory = None
    if cfg.memory_backend == "pinecone":
        if not PineconeMemory:
            internal_print("Error: Pinecone is not installed. Please install pinecone"
                  " to use Pinecone as a memory backend.")
        else:
            memory = PineconeMemory(cfg)
            if init:
                memory.clear()
    elif cfg.memory_backend == "redis":
        if not RedisMemory:
            internal_print("Error: Redis is not installed. Please install redis-py to"
                  " use Redis as a memory backend.")
        else:
            memory = RedisMemory(cfg)

    if memory is None:
        memory = LocalCache(cfg)
        if init:
            memory.clear()
    return memory


__all__ = [
    "get_memory",
    "LocalCache",
    "RedisMemory",
    "PineconeMemory",
]
