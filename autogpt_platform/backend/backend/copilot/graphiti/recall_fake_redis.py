"""The recall tests' in-memory Redis. Not collected by pytest."""


class FakeRedis:
    """Honours SET NX like Redis, so ``record_memory_hit`` counts correctly,
    and keeps hashes for the forget stash (``recall_stash``)."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def get(self, key: str) -> bytes | None:
        return str(self.values[key]).encode() if key in self.values else None

    async def set(self, key: str, value: int, **kwargs: object) -> bool:
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = int(value)
        return True

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return True

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> int:
        stored = self.hashes.get(key, {})
        return len([f for f in fields if stored.pop(f, None) is not None])

    def pipeline(self, transaction: bool = True) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """``FakeRedis``'s transaction: queued hash writes applied on execute."""

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.writes: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def hset(self, key: str, mapping: dict[str, str]) -> "FakePipeline":
        self.writes.append((key, mapping))
        return self

    def expire(self, key: str, ttl_seconds: int) -> "FakePipeline":
        return self

    async def execute(self) -> list[object]:
        for key, mapping in self.writes:
            self.redis.hashes.setdefault(key, {}).update(mapping)
        return []
