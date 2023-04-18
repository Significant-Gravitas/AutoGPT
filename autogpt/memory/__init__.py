from autogpt.memory.local import LocalCache
from autogpt.memory.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ["local", "no_memory"]

try:
    from autogpt.memory.pinecone import PineconeMemory

    supported_memory.append("pinecone")
except ImportError:
    # print("Pinecone not installed. Skipping import.")
    PineconeMemory = None

try:
    from autogpt.memory.redismem import RedisMemory

    supported_memory.append("redis")
except ImportError:
    # print("Redis not installed. Skipping import.")
    RedisMemory = None

try:
    from autogpt.memory.weaviate import WeaviateMemory

    supported_memory.append("weaviate")
except ImportError:
    # print("Weaviate not installed. Skipping import.")
    WeaviateMemory = None

try:
    from autogpt.memory.milvus import MilvusMemory

    supported_memory.append("milvus")
except ImportError:
    # print("pymilvus not installed. Skipping import.")
    MilvusMemory = None


def get_memory(cfg, init=False):
    memory = None
    # memory_backend convert to lowercase
    memory_backend = cfg.memory_backend.lower()
    if memory_backend == "pinecone":
        if not PineconeMemory:
            print(
                "Error: Pinecone is not installed."
                "Please install pinecone to use Pinecone as a memory backend."
            )
        else:
            memory = PineconeMemory(cfg)
            if init:
                memory.clear()
    elif memory_backend == "redis":
        if not RedisMemory:
            print(
                "Error: Redis is not installed."
                "Please install redis-py to use Redis as a memory backend."
            )
        else:
            memory = RedisMemory(cfg)
    elif memory_backend == "weaviate":
        if not WeaviateMemory:
            print(
                "Error: Weaviate is not installed."
                "Please install weaviate-client to use Weaviate as a memory backend."
            )
        else:
            memory = WeaviateMemory(cfg)
    elif memory_backend == "milvus":
        if not MilvusMemory:
            print(
                "Error: Milvus is not installed."
                "Please install pymilvus to use Milvus as a memory backend."
            )
        else:
            memory = MilvusMemory(cfg)
    elif memory_backend == "no_memory":
        memory = NoMemory(cfg)

    if memory is None:
        memory = LocalCache(cfg)
        if init:
            memory.clear()
    return memory


def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "LocalCache",
    "NoMemory",
    "PineconeMemory",
    "RedisMemory",
    "MilvusMemory",
    "WeaviateMemory",
    "get_memory",
    "get_supported_memory_backends",
]
