from binascii import crc_hqx

from redis.typing import EncodedT

# Redis Cluster's key space is divided into 16384 slots.
# For more information see: https://github.com/redis/redis/issues/2576
REDIS_CLUSTER_HASH_SLOTS = 16384

__all__ = ["key_slot", "REDIS_CLUSTER_HASH_SLOTS"]


def key_slot(key: EncodedT, bucket: int = REDIS_CLUSTER_HASH_SLOTS) -> int:
    """Calculate key slot for a given key.
    See Keys distribution model in https://redis.io/topics/cluster-spec
    :param key - bytes
    :param bucket - int
    """
    start = key.find(b"{")
    if start > -1:
        end = key.find(b"}", start + 1)
        if end > -1 and end != start + 1:
            key = key[start + 1 : end]
    return crc_hqx(key, 0) % bucket
