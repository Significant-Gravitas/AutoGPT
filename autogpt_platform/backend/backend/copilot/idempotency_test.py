"""Tests for the per-session idempotency-key dedup helpers."""

from typing import Any

import pytest

from backend.copilot import idempotency as id_module


class _FakeAsyncRedis:
    """Minimal async-redis stand-in covering only the surface this module uses."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.fail = False

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if self.fail:
            raise RuntimeError("redis down")
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise RuntimeError("redis down")
        return self.store.get(key)


class _FakeSyncRedis:
    def __init__(self, store: dict[str, str], fail: bool = False) -> None:
        self.store = store
        self.fail = fail

    def get(self, key: str) -> str | None:
        if self.fail:
            raise RuntimeError("redis down")
        return self.store.get(key)


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeAsyncRedis:
    fake = _FakeAsyncRedis()

    async def _get_redis_async() -> _FakeAsyncRedis:
        return fake

    monkeypatch.setattr(id_module, "get_redis_async", _get_redis_async)
    monkeypatch.setattr(
        id_module, "get_redis", lambda: _FakeSyncRedis(fake.store, fake.fail)
    )
    return fake


@pytest.mark.asyncio
async def test_first_claim_returns_none(fake_redis: _FakeAsyncRedis) -> None:
    assert (
        await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
        is None
    )
    # Stored under the namespaced key for later read-only lookups.
    assert fake_redis.store["idempotency:stream:sess1:key-A"] == "turn-1"


@pytest.mark.asyncio
async def test_second_claim_returns_existing_turn(
    fake_redis: _FakeAsyncRedis,
) -> None:
    await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
    existing = await id_module.claim_stream_idempotency_key(
        "sess1", "key-A", turn_id="turn-2"
    )
    assert existing == "turn-1"
    # First claim wins; the store does not get overwritten.
    assert fake_redis.store["idempotency:stream:sess1:key-A"] == "turn-1"


@pytest.mark.asyncio
async def test_distinct_keys_in_same_session_do_not_collide(
    fake_redis: _FakeAsyncRedis,
) -> None:
    assert (
        await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
        is None
    )
    assert (
        await id_module.claim_stream_idempotency_key("sess1", "key-B", turn_id="turn-2")
        is None
    )


@pytest.mark.asyncio
async def test_same_key_in_different_sessions_do_not_collide(
    fake_redis: _FakeAsyncRedis,
) -> None:
    assert (
        await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
        is None
    )
    assert (
        await id_module.claim_stream_idempotency_key("sess2", "key-A", turn_id="turn-2")
        is None
    )


@pytest.mark.asyncio
async def test_claim_fails_open_on_redis_error(
    fake_redis: _FakeAsyncRedis,
) -> None:
    fake_redis.fail = True
    assert (
        await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
        is None
    )


@pytest.mark.asyncio
async def test_get_returns_existing_or_none(
    fake_redis: _FakeAsyncRedis,
) -> None:
    assert (await id_module.get_claimed_turn_id("sess1", "key-A")) is None
    await id_module.claim_stream_idempotency_key("sess1", "key-A", turn_id="turn-1")
    assert await id_module.get_claimed_turn_id("sess1", "key-A") == "turn-1"


@pytest.mark.asyncio
async def test_get_fails_open_on_redis_error(
    fake_redis: _FakeAsyncRedis,
) -> None:
    fake_redis.fail = True
    assert (await id_module.get_claimed_turn_id("sess1", "key-A")) is None


def test_sync_get_returns_existing_or_none(
    fake_redis: _FakeAsyncRedis,
) -> None:
    fake_redis.store["idempotency:stream:sess1:key-A"] = "turn-1"
    assert id_module.sync_get_claimed_turn_id("sess1", "key-A") == "turn-1"
    assert id_module.sync_get_claimed_turn_id("sess1", "key-missing") is None


def test_sync_get_fails_open_on_redis_error(
    fake_redis: _FakeAsyncRedis,
) -> None:
    fake_redis.fail = True
    assert id_module.sync_get_claimed_turn_id("sess1", "key-A") is None


@pytest.mark.asyncio
async def test_claim_after_expiry_treated_as_fresh(
    fake_redis: _FakeAsyncRedis,
) -> None:
    """If SET-NX fails but the GET fallback returns None (key expired in
    the millisecond gap), treat it as a fresh claim — the original turn
    is gone, no parallel turn risk."""

    class _ExpiringRedis(_FakeAsyncRedis):
        async def set(
            self,
            key: str,
            value: str,
            *,
            ex: int | None = None,
            nx: bool = False,
        ) -> bool | None:
            return None  # SET NX returns None on collision

        async def get(self, key: str) -> str | None:
            return None  # And the key has since expired

    expiring = _ExpiringRedis()

    async def _get_redis_async() -> Any:
        return expiring

    import pytest as _p

    with _p.MonkeyPatch.context() as mp:
        mp.setattr(id_module, "get_redis_async", _get_redis_async)
        assert (
            await id_module.claim_stream_idempotency_key(
                "sess1", "key-A", turn_id="turn-1"
            )
            is None
        )
