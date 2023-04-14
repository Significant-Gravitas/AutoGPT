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


def get_memory(cfg, init=False):
    """
    get_memory returns a memory object based on the specified memory backend in
    the configuration.
    
    :param cfg: a configuration object that contains information about the memory
    backend to be used and other relevant parameters
    :param init: A boolean parameter that indicates whether to initialize the memory
    or not. If set to True, the memory will be cleared, defaults to False (optional)
    :return: an instance of a memory object based on the configuration provided. The
    type of memory object returned depends on the value of the `memory_backend`
    attribute in the configuration. If `memory_backend` is set to "pinecone", a
    `PineconeMemory` object is returned. If it is set to "redis", a `RedisMemory`
    object is returned. By default, a `LocalCache` object is returned.
    """
    memory = None
    if cfg.memory_backend == "pinecone":
        if not PineconeMemory:
            print("Error: Pinecone is not installed. Please install pinecone"
                  " to use Pinecone as a memory backend.")
        else:
            memory = PineconeMemory(cfg)
            if init:
                memory.clear()
    elif cfg.memory_backend == "redis":
        if not RedisMemory:
            print("Error: Redis is not installed. Please install redis-py to"
                  " use Redis as a memory backend.")
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
