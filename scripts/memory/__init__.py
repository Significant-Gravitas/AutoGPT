from memory.local import LocalCache
from memory.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ['local']

try:
    from memory.redismem import RedisMemory
    supported_memory.append('redis')
except ImportError:
    print("Redis 未安装。跳过导入。")
    RedisMemory = None

try:
    from memory.pinecone import PineconeMemory
    supported_memory.append('pinecone')
except ImportError:
    print("Pinecone 未安装。跳过导入。")
    PineconeMemory = None


def get_memory(cfg, init=False):
    memory = None
    if cfg.memory_backend == "pinecone":
        if not PineconeMemory:
            print("错误: Pinecone 未安装。请安装 pinecone"
                   " 使用 Pinecone 作为内存后端。")
        else:
            memory = PineconeMemory(cfg)
            if init:
                memory.clear()
    elif cfg.memory_backend == "redis":
        if not RedisMemory:
            print("错误: Redis 未安装。请安装 redis-py"
                   " 使用 Redis 作为内存后端。")
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


__all__ = [
    "get_memory",
    "LocalCache",
    "RedisMemory",
    "PineconeMemory",
    "NoMemory"
]
