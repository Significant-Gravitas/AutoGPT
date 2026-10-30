"""The recall tests' in-memory Redis. Not collected by pytest."""

from backend.copilot.dream.locks import EXTEND_SCRIPT, UNLOCK_SCRIPT


class FakeRedis:
    """Honours SET NX like Redis, for the memory-hit counter
    (``record_memory_hit``) and the graph write lock (``scope_lock.py``),
    and runs the lock's compare-and-delete and compare-and-extend scripts.
    Values are stored as strings, as the ``decode_responses`` client returns
    them; TTLs are recorded, not enforced."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        # The graphs noted as holding a pending dream record
        # (``provenance_pending.py``), by set.
        self.sets: dict[str, set[str]] = {}

    async def sadd(self, name: str, *values: object) -> int:
        bucket = self.sets.setdefault(name, set())
        added = {str(value) for value in values} - bucket
        bucket |= added
        return len(added)

    async def srem(self, name: str, *values: object) -> int:
        bucket = self.sets.get(name, set())
        removed = {str(value) for value in values} & bucket
        bucket -= removed
        return len(removed)

    async def srandmember(self, name: str, number: int) -> list[str]:
        return sorted(self.sets.get(name, set()))[:number]

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: object,
        *,
        nx: bool = False,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool | None:
        if nx and key in self.values:
            return None
        self.values[key] = str(value)
        self.ttls[key] = ex if ex is not None else (px or 0) // 1000
        return True

    async def incr(self, key: str) -> int:
        self.values[key] = str(int(self.values.get(key, "0")) + 1)
        return int(self.values[key])

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        self.ttls[key] = int(ttl_seconds)
        return key in self.values

    async def eval(self, script: str, numkeys: int, key: str, *args: object) -> int:
        token, *rest = (str(arg) for arg in args)
        if self.values.get(key) != token:
            return 0
        if script == UNLOCK_SCRIPT:
            del self.values[key]
            return 1
        if script == EXTEND_SCRIPT:
            self.ttls[key] = int(rest[0])
            return 1
        raise NotImplementedError("FakeRedis runs only the lock scripts")
