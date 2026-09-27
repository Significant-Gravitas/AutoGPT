"""Owner-checked operations on a token lock.

A token lock is a string key whose value is its holder's token. Releasing or
renewing one has to compare the token and act in one step: a holder whose lock
lapsed and was re-acquired would otherwise free or stretch the new holder's.
Each is a single-key script, so it routes on Redis Cluster.

The bodies are Redis Lua scripts, written as Python and compiled by
redis-lua-py; Python never runs them. Call one with the client first:
``await delete_if_owner(redis, key=key, token=token)``.
"""

from redis_lua_py import Key, redis, script


@script
def delete_if_owner(key: Key, token: str) -> int:
    """``DEL`` *key* if it still holds *token*: 1 if deleted, else 0."""
    if redis.get(key) == token:
        return redis.delete(key)
    return 0


@script
def expire_if_owner(key: Key, token: str, seconds: int) -> int:
    """``EXPIRE`` *key* if it still holds *token*: 1 if renewed, else 0."""
    if redis.get(key) == token:
        return redis.expire(key, seconds)
    return 0


@script
def pexpire_if_owner(key: Key, token: str, milliseconds: int) -> int:
    """``PEXPIRE`` *key* if it still holds *token*: 1 if renewed, else 0."""
    if redis.get(key) == token:
        return redis.pexpire(key, milliseconds)
    return 0
