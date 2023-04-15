from autogpt.memory.local import LocalCache
from autogpt.memory.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ["local", "no_memory"]

try:
    from autogpt.memory.redismem import RedisMemory

    supported_memory.append("redis")
except ImportError:
    print("Redis not installed. Skipping import.")
    RedisMemory = None

try:
    from autogpt.memory.pinecone import PineconeMemory

    supported_memory.append("pinecone")
except ImportError:
    print("Pinecone not installed. Skipping import.")
    PineconeMemory = None


def get_memory(cfg, init=False):
    memory = None
    if cfg.memory_backend == "pinecone":
        if not PineconeMemory:
            print(
                "Error: Pinecone is not installed. Please install pinecone"
                " to use Pinecone as a memory backend."
            )
        else:
            memory = PineconeMemory(cfg)
            if init:
                memory.clear()
    elif cfg.memory_backend == "redis":
        if not RedisMemory:
            print(
                "Error: Redis is not installed. Please install redis-py to"
                " use Redis as a memory backend."
            )
        else:
            memory = RedisMemory(cfg)
    elif cfg.memory_backend == "no_memory":
        memory = NoMemory(cfg)

    if memory is None:
        memory = LocalCache(cfg)
        if init:
            memory.clear()
    return memory


def get_supported_memory_backends():
    return supported_memory


__all__ = ["get_memory", "LocalCache", "RedisMemory", "PineconeMemory", "NoMemory"]
