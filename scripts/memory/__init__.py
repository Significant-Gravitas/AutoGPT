from memory.local import LocalCache

try:
    from memory.redismem import RedisMemory
except ImportError:
    print("Redis not installed. Skipping import.")
    RedisMemory = None


def get_memory(cfg, init=False):
    memory = None
    if cfg.memory_backend == "redis":
        if RedisMemory:
            memory = RedisMemory(cfg)
        else:
            print(
                "Error: Redis is not installed. Please install redis-py to use Redis as a memory backend."
            )
    if memory is None:
        memory = LocalCache(cfg)
        if init:
            memory.clear()
    return memory


__all__ = [
    "get_memory",
    "LocalCache",
    "RedisMemory",
]
