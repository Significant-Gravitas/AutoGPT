"""Unit tests for :mod:`backend.data.redis_helpers`.

Uses a minimal in-memory fake Redis that only implements the surface
exercised by the helpers: pipeline(transaction=True) with
incr/expire/rpush/ltrim/llen, and eval() for the CAS helper.
"""

from typing import Any

import pytest

from backend.data.redis_helpers import (
    capped_rpush,
    hash_compare_and_set,
    incr_with_ttl,
    incr_with_ttl_sync,
)

# ── Fake Redis + pipeline ──────────────────────────────────────────────


class _Fake:
    """Async-only fake.  Enough for ``incr_with_ttl`` + ``capped_rpush`` + CAS."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int, bool]] = []

    # --- primitives ---
    async def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key: str, seconds: int, nx: bool = False) -> int:
        self.expire_calls.append((key, seconds, nx))
        if nx and key in self.ttls:
            return 0
        self.ttls[key] = seconds
        return 1

    async def rpush(self, key: str, *values: Any) -> int:
        self.lists.setdefault(key, []).extend(str(v) for v in values)
        return len(self.lists[key])

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        lst = self.lists.get(key, [])
        if stop == -1:
            self.lists[key] = lst[start:]
        else:
            self.lists[key] = lst[start : stop + 1]

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        # Shim for hash-CAS only.
        key, field, expected, new = args[0], args[1], args[2], args[3]
        h = self.hashes.setdefault(key, {})
        if h.get(field) == expected:
            h[field] = new
            return 1
        return 0

    # --- pipeline ---
    def pipeline(self, transaction: bool = True) -> "_FakePipe":
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, parent: _Fake) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def incr(self, key: str) -> "_FakePipe":
        self._ops.append(("incr", (key,), {}))
        return self

    def expire(self, key: str, seconds: int, **kw: Any) -> "_FakePipe":
        self._ops.append(("expire", (key, seconds), kw))
        return self

    def rpush(self, key: str, value: Any) -> "_FakePipe":
        self._ops.append(("rpush", (key, value), {}))
        return self

    def ltrim(self, key: str, start: int, stop: int) -> "_FakePipe":
        self._ops.append(("ltrim", (key, start, stop), {}))
        return self

    def llen(self, key: str) -> "_FakePipe":
        self._ops.append(("llen", (key,), {}))
        return self

    async def execute(self) -> list[Any]:
        out: list[Any] = []
        for name, args, kw in self._ops:
            out.append(await getattr(self._parent, name)(*args, **kw))
        return out


class _SyncFake:
    """Sync fake for :func:`incr_with_ttl_sync`."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int, bool]] = []

    def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def expire(self, key: str, seconds: int, nx: bool = False) -> int:
        self.expire_calls.append((key, seconds, nx))
        if nx and key in self.ttls:
            return 0
        self.ttls[key] = seconds
        return 1

    def pipeline(self, transaction: bool = True) -> "_SyncPipe":
        return _SyncPipe(self)


class _SyncPipe:
    def __init__(self, parent: _SyncFake) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def incr(self, key: str) -> "_SyncPipe":
        self._ops.append(("incr", (key,), {}))
        return self

    def expire(self, key: str, seconds: int, **kw: Any) -> "_SyncPipe":
        self._ops.append(("expire", (key, seconds), kw))
        return self

    def execute(self) -> list[Any]:
        return [
            getattr(self._parent, name)(*args, **kw) for name, args, kw in self._ops
        ]


# ── incr_with_ttl ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_incr_with_ttl_returns_count_and_sets_ttl_once() -> None:
    """Fixed-window: TTL is set only on first bump, subsequent calls keep it."""
    r = _Fake()
    assert await incr_with_ttl(r, "k", 60) == 1  # type: ignore[arg-type]
    assert await incr_with_ttl(r, "k", 60) == 2  # type: ignore[arg-type]
    assert await incr_with_ttl(r, "k", 60) == 3  # type: ignore[arg-type]
    # EXPIRE is always called, but with nx=True so only first succeeds.
    assert all(nx for _, _, nx in r.expire_calls)
    assert r.expire_calls == [("k", 60, True)] * 3


@pytest.mark.asyncio
async def test_incr_with_ttl_sliding_window_refreshes_every_bump() -> None:
    r = _Fake()
    await incr_with_ttl(r, "k", 10, reset_ttl_on_bump=True)  # type: ignore[arg-type]
    await incr_with_ttl(r, "k", 20, reset_ttl_on_bump=True)  # type: ignore[arg-type]
    # nx=False on every call → TTL is refreshed each bump.
    assert r.expire_calls == [("k", 10, False), ("k", 20, False)]
    assert r.ttls["k"] == 20


def test_incr_with_ttl_sync_behaves_the_same() -> None:
    r = _SyncFake()
    assert incr_with_ttl_sync(r, "k", 5) == 1  # type: ignore[arg-type]
    assert incr_with_ttl_sync(r, "k", 5) == 2  # type: ignore[arg-type]
    assert r.expire_calls == [("k", 5, True), ("k", 5, True)]


# ── capped_rpush ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capped_rpush_returns_length_and_trims() -> None:
    r = _Fake()
    for i in range(5):
        length = await capped_rpush(
            r, "buf", f"item-{i}", max_len=3, ttl_seconds=30  # type: ignore[arg-type]
        )
    # After 5 pushes capped at 3, only the newest 3 remain.
    assert length == 3
    assert r.lists["buf"] == ["item-2", "item-3", "item-4"]


@pytest.mark.asyncio
async def test_capped_rpush_first_push_returns_one() -> None:
    r = _Fake()
    length = await capped_rpush(r, "buf", "only", max_len=10, ttl_seconds=60)  # type: ignore[arg-type]
    assert length == 1
    assert r.lists["buf"] == ["only"]


# ── hash_compare_and_set ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hash_cas_swaps_when_expected_matches() -> None:
    r = _Fake()
    r.hashes["meta"] = {"status": "running"}
    swapped = await hash_compare_and_set(
        r, "meta", "status", expected="running", new="completed"  # type: ignore[arg-type]
    )
    assert swapped is True
    assert r.hashes["meta"]["status"] == "completed"


@pytest.mark.asyncio
async def test_hash_cas_no_swap_when_expected_differs() -> None:
    r = _Fake()
    r.hashes["meta"] = {"status": "completed"}
    swapped = await hash_compare_and_set(
        r, "meta", "status", expected="running", new="failed"  # type: ignore[arg-type]
    )
    assert swapped is False
    assert r.hashes["meta"]["status"] == "completed"
