from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Awaitable, Callable
from typing import Any


class _Pipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.calls: list[Callable[[], Awaitable[Any]]] = []

    def hset(
        self,
        key: str,
        field: str | None = None,
        value: str | None = None,
        *,
        mapping: dict[str, Any] | None = None,
    ) -> _Pipeline:
        self.calls.append(lambda: self.redis.hset(key, field, value, mapping=mapping))
        return self

    def expire(self, key: str, seconds: int) -> _Pipeline:
        self.calls.append(lambda: self.redis.expire(key, seconds))
        return self

    def xadd(
        self,
        key: str,
        fields: dict[str, str],
        *,
        maxlen: int,
        approximate: bool,
    ) -> _Pipeline:
        self.calls.append(
            lambda: self.redis.xadd(key, fields, maxlen=maxlen, approximate=approximate)
        )
        return self

    def incrby(self, key: str, amount: int) -> _Pipeline:
        self.calls.append(lambda: self.redis.incrby(key, amount))
        return self

    def decrby(self, key: str, amount: int) -> _Pipeline:
        self.calls.append(lambda: self.redis.decrby(key, amount))
        return self

    def xdel(self, key: str, *message_ids: str) -> _Pipeline:
        self.calls.append(lambda: self.redis.xdel(key, *message_ids))
        return self

    def zadd(self, key: str, mapping: dict[str, float]) -> _Pipeline:
        self.calls.append(lambda: self.redis.zadd(key, mapping))
        return self

    def zremrangebyrank(self, key: str, start: int, stop: int) -> _Pipeline:
        self.calls.append(lambda: self.redis.zremrangebyrank(key, start, stop))
        return self

    def zremrangebyscore(
        self, key: str, minimum: str | float, maximum: str | float
    ) -> _Pipeline:
        self.calls.append(lambda: self.redis.zremrangebyscore(key, minimum, maximum))
        return self

    def zrem(self, key: str, *members: str) -> _Pipeline:
        self.calls.append(lambda: self.redis.zrem(key, *members))
        return self

    async def execute(self) -> list[Any]:
        return [await call() for call in self.calls]


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.strings: dict[str, int] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.expirations: dict[str, int] = {}
        self.pipeline_transactions: list[bool] = []
        self.sequence = 0
        self._eval_lock = asyncio.Lock()

    def pipeline(self, transaction: bool = True) -> _Pipeline:
        self.pipeline_transactions.append(transaction)
        return _Pipeline(self)

    async def get(self, key: str) -> str | None:
        value = self.strings.get(key)
        return str(value) if value is not None else None

    async def incrby(self, key: str, amount: int) -> int:
        self.strings[key] = self.strings.get(key, 0) + amount
        return self.strings[key]

    async def decrby(self, key: str, amount: int) -> int:
        return await self.incrby(key, -amount)

    async def hset(
        self,
        key: str,
        field: str | None = None,
        value: str | None = None,
        *,
        mapping: dict[str, Any] | None = None,
    ) -> int:
        target = self.hashes.setdefault(key, {})
        if mapping is not None:
            target.update({str(k): str(v) for k, v in mapping.items()})
        elif field is not None and value is not None:
            target[str(field)] = str(value)
        return 1

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hmget(self, key: str, fields: list[str]) -> list[str | None]:
        return [self.hashes.get(key, {}).get(field) for field in fields]

    async def hkeys(self, key: str) -> list[str]:
        return list(self.hashes.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> int:
        target = self.hashes.get(key, {})
        return sum(target.pop(field, None) is not None for field in fields)

    async def expire(self, key: str, seconds: int) -> bool:
        self.expirations[key] = seconds
        return True

    async def eval(self, script: str, number_of_keys: int, *keys_and_args: str) -> int:
        keys = keys_and_args[:number_of_keys]
        args = keys_and_args[number_of_keys:]
        async with self._eval_lock:
            if "LOCAL_EXECUTOR_STORE_RECORDING_STATE" in script:
                state_key, order_key = keys
                recording_id, serialized_state, score, max_entries, ttl = args
                self.hashes.setdefault(state_key, {})[recording_id] = serialized_state
                order = self.sorted_sets.setdefault(order_key, {})
                order[recording_id] = float(score)
                ordered_ids = sorted(order, key=lambda item: (order[item], item))
                stale_ids = ordered_ids[: max(0, len(order) - int(max_entries))]
                for stale_id in stale_ids:
                    order.pop(stale_id, None)
                    self.hashes[state_key].pop(stale_id, None)
                self.expirations[state_key] = int(ttl)
                self.expirations[order_key] = int(ttl)
                return len(stale_ids)
            if "LOCAL_EXECUTOR_BOUNDED_APPEND" in script:
                key, counter_key = keys
                size, max_entries, max_bytes, _ttl, *flat_fields = args
                if len(self.streams.get(key, [])) >= int(
                    max_entries
                ) or self.strings.get(counter_key, 0) + int(size) > int(max_bytes):
                    return 0
                fields = dict(zip(flat_fields[::2], flat_fields[1::2]))
                await self.xadd(
                    key,
                    fields,
                    maxlen=int(max_entries),
                    approximate=False,
                )
                self.strings[counter_key] = self.strings.get(counter_key, 0) + int(size)
                return 1
            if "LOCAL_EXECUTOR_ACKNOWLEDGE" in script:
                key, counter_key = keys
                message_id, size, _ttl = args
                removed = await self.xdel(key, message_id)
                if not removed:
                    return 0
                self.strings[counter_key] = max(
                    0, self.strings.get(counter_key, 0) - int(size)
                )
                return 1

            key = keys[0]
            target = self.hashes.get(key)
            if target is None or target.get("connection_id") != args[0]:
                return 0
            if "HSET" in script:
                target["expires_at"] = args[1]
            else:
                self.hashes.pop(key, None)
            return 1

    async def xadd(
        self,
        key: str,
        fields: dict[str, str],
        *,
        maxlen: int,
        approximate: bool,
    ) -> str:
        self.sequence += 1
        message_id = f"{self.sequence}-0"
        self.streams.setdefault(key, []).append((message_id, dict(fields)))
        self.streams[key] = self.streams[key][-maxlen:]
        return message_id

    async def xread(
        self,
        *,
        streams: dict[str, str],
        block: int,
        count: int,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        key, cursor = next(iter(streams.items()))
        deadline = asyncio.get_running_loop().time() + block / 1_000
        while True:
            available = [
                item
                for item in self.streams.get(key, [])
                if int(item[0].split("-", 1)[0]) > int(cursor.split("-", 1)[0])
            ][:count]
            if available:
                return [(key, available)]
            if asyncio.get_running_loop().time() >= deadline:
                return []
            await asyncio.sleep(0.005)

    async def xrevrange(self, key: str, *, count: int) -> list[Any]:
        return list(reversed(self.streams.get(key, [])))[:count]

    async def xlen(self, key: str) -> int:
        return len(self.streams.get(key, []))

    async def xdel(self, key: str, *message_ids: str) -> int:
        existing = self.streams.get(key, [])
        retained = [item for item in existing if item[0] not in message_ids]
        self.streams[key] = retained
        return len(existing) - len(retained)

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.sorted_sets.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        scores = self.sorted_sets.get(key, {})
        values = sorted(scores, key=lambda value: scores[value], reverse=True)
        return values[start:] if stop == -1 else values[start : stop + 1]

    async def zrange(self, key: str, start: int, stop: int) -> list[str]:
        scores = self.sorted_sets.get(key, {})
        values = sorted(scores, key=lambda value: scores[value])
        return values[start:] if stop == -1 else values[start : stop + 1]

    async def zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        values = await self.zrange(key, 0, -1)
        length = len(values)
        normalized_start = max(0, length + start) if start < 0 else start
        normalized_stop = length + stop if stop < 0 else stop
        normalized_stop = min(length - 1, normalized_stop)
        selected = (
            values[normalized_start : normalized_stop + 1]
            if normalized_start <= normalized_stop
            else []
        )
        for value in selected:
            self.sorted_sets.get(key, {}).pop(value, None)
        return len(selected)

    async def zremrangebyscore(
        self, key: str, minimum: str | float, maximum: str | float
    ) -> int:
        lower = float("-inf") if minimum == "-inf" else float(minimum)
        upper = float("inf") if maximum == "+inf" else float(maximum)
        target = self.sorted_sets.get(key, {})
        selected = [
            member for member, score in target.items() if lower <= score <= upper
        ]
        for member in selected:
            target.pop(member, None)
        return len(selected)

    async def zrem(self, key: str, *members: str) -> int:
        target = self.sorted_sets.get(key, {})
        return sum(target.pop(member, None) is not None for member in members)

    async def scan_iter(self, *, match: str):
        for key in list(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key


class FakeWebSocket:
    def __init__(self) -> None:
        self.inbound: asyncio.Queue[str | None] = asyncio.Queue()
        self.outbound: asyncio.Queue[str] = asyncio.Queue()
        self.closed = asyncio.Event()
        self.close_code: int | None = None

    async def send_text(self, data: str) -> None:
        await self.outbound.put(data)

    async def iter_text(self):
        while True:
            value = await self.inbound.get()
            if value is None:
                return
            yield value

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.closed.set()
        await self.inbound.put(None)
