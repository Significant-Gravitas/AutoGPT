import asyncio
import pickle
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.util import cache as cache_module
from backend.util import subscription_tiers
from backend.util.cache import _make_hashable_key, _make_redis_key, _sign_payload
from backend.util.service import UnhealthyServiceError


class _MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.fail_reads = False
        self.fail_writes = False
        self.fail_delete = False
        self.delete_calls = 0

    def get(self, key: str) -> bytes | None:
        if self.fail_reads:
            raise ConnectionError("Redis unavailable")
        return self.values.get(key)

    def setex(self, key: str, _ttl: int, value: bytes) -> bool:
        if self.fail_writes:
            raise ConnectionError("Redis unavailable")
        self.values[key] = value
        return True

    def delete(self, key: str) -> int:
        self.delete_calls += 1
        if self.fail_delete:
            raise ConnectionError("Redis unavailable")
        return int(self.values.pop(key, None) is not None)


@pytest.fixture
def tier_cache(monkeypatch: pytest.MonkeyPatch) -> _MemoryRedis:
    redis = _MemoryRedis()
    monkeypatch.setattr(cache_module, "_get_redis", lambda: redis)
    monkeypatch.setattr(cache_module, "_shared_cache_redis", None)
    monkeypatch.setattr(cache_module, "_shared_cache_redis_retry_at", 0.0)
    return redis


def _database_client(*tiers: SubscriptionTier | Exception) -> MagicMock:
    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=tiers)
    return client


def test_subscription_tier_order_covers_every_tier_once():
    assert len(subscription_tiers.SUBSCRIPTION_TIER_ORDER) == len(
        set(subscription_tiers.SUBSCRIPTION_TIER_ORDER)
    )
    assert set(subscription_tiers.SUBSCRIPTION_TIER_ORDER) == set(SubscriptionTier)


def test_subscription_tier_at_least_uses_canonical_order():
    assert subscription_tiers.subscription_tier_at_least(
        SubscriptionTier.MAX, SubscriptionTier.MAX
    )
    assert subscription_tiers.subscription_tier_at_least(
        SubscriptionTier.ENTERPRISE, SubscriptionTier.MAX
    )
    assert not subscription_tiers.subscription_tier_at_least(
        SubscriptionTier.PRO, SubscriptionTier.MAX
    )


@pytest.mark.asyncio
async def test_successful_lookup_is_cached_for_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    client = _database_client(SubscriptionTier.MAX)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    results = await asyncio.gather(
        *(subscription_tiers.get_user_subscription_tier("user-1") for _ in range(8))
    )

    assert results == [SubscriptionTier.MAX] * 8
    client.get_user_subscription_tier.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_missing_users_and_outages_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    client = _database_client(
        ValueError("missing"),
        UnhealthyServiceError("Database Manager unavailable"),
        SubscriptionTier.BUSINESS,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    with pytest.raises(subscription_tiers.SubscriptionTierUserNotFoundError):
        await subscription_tiers.get_user_subscription_tier("user-1")
    with pytest.raises(UnhealthyServiceError):
        await subscription_tiers.get_user_subscription_tier("user-1")
    assert (
        await subscription_tiers.get_user_subscription_tier("user-1")
        == SubscriptionTier.BUSINESS
    )
    assert client.get_user_subscription_tier.await_count == 3


@pytest.mark.asyncio
async def test_unexpected_database_error_propagates_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    client = _database_client(RuntimeError("database failed"), SubscriptionTier.MAX)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="database failed"):
        await subscription_tiers.get_user_subscription_tier("user-1")
    assert (
        await subscription_tiers.get_user_subscription_tier("user-1")
        == SubscriptionTier.MAX
    )
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_redis_outage_uses_short_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    tier_cache.fail_reads = True
    tier_cache.fail_writes = True
    client = _database_client(SubscriptionTier.MAX)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    assert (
        await subscription_tiers.get_user_subscription_tier("user-1")
        == SubscriptionTier.MAX
    )
    assert (
        await subscription_tiers.get_user_subscription_tier("user-1")
        == SubscriptionTier.MAX
    )
    assert client.get_user_subscription_tier.await_count == 1


@pytest.mark.asyncio
async def test_redis_outage_local_fallback_expires_after_one_second(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    clock = [100.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])
    tier_cache.fail_reads = True
    tier_cache.fail_writes = True
    client = _database_client(SubscriptionTier.MAX, SubscriptionTier.PRO)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    assert (
        await subscription_tiers.get_user_subscription_tier("fallback-expiry-user")
        == SubscriptionTier.MAX
    )
    clock[0] = 101.1
    assert (
        await subscription_tiers.get_user_subscription_tier("fallback-expiry-user")
        == SubscriptionTier.PRO
    )
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_invalidation_is_user_specific(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    values = {
        "changed": [SubscriptionTier.MAX, SubscriptionTier.PRO],
        "stable": [SubscriptionTier.BUSINESS],
    }

    async def read_tier(user_id: str) -> SubscriptionTier:
        return values[user_id].pop(0)

    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=read_tier)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    assert (
        await subscription_tiers.get_user_subscription_tier("changed")
        == SubscriptionTier.MAX
    )
    assert (
        await subscription_tiers.get_user_subscription_tier("stable")
        == SubscriptionTier.BUSINESS
    )

    subscription_tiers.invalidate_user_subscription_tier("changed")

    assert (
        await subscription_tiers.get_user_subscription_tier("changed")
        == SubscriptionTier.PRO
    )
    assert (
        await subscription_tiers.get_user_subscription_tier("stable")
        == SubscriptionTier.BUSINESS
    )
    assert client.get_user_subscription_tier.await_count == 3


@pytest.mark.asyncio
async def test_existing_cache_namespace_and_enum_payload_are_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tier_cache: _MemoryRedis,
):
    from enum import Enum

    from backend.copilot import rate_limit

    current_tier_enum = rate_limit.SubscriptionTier
    legacy_tier_enum = Enum(
        "SubscriptionTier",
        {"MAX": "MAX"},
        type=str,
        module=rate_limit.__name__,
    )
    monkeypatch.setattr(rate_limit, "SubscriptionTier", legacy_tier_enum)
    try:
        legacy_payload = pickle.dumps(legacy_tier_enum.MAX)
    finally:
        monkeypatch.setattr(rate_limit, "SubscriptionTier", current_tier_enum)

    key = _make_hashable_key(("user-1",), {})
    redis_key = _make_redis_key(key, "_fetch_user_tier")
    tier_cache.values[redis_key] = _sign_payload(legacy_payload)
    get_db = MagicMock()
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        get_db,
    )

    assert (
        await subscription_tiers.get_user_subscription_tier("user-1")
        == SubscriptionTier.MAX
    )
    assert redis_key.startswith("cache:_fetch_user_tier:")
    get_db.assert_not_called()


def test_invalidation_is_best_effort_when_redis_is_unavailable(
    tier_cache: _MemoryRedis,
    caplog: pytest.LogCaptureFixture,
):
    tier_cache.fail_delete = True

    with caplog.at_level("WARNING"):
        subscription_tiers.invalidate_user_subscription_tier("user-1")

    assert tier_cache.delete_calls == 1
    assert "Failed to invalidate subscription tier cache" in caplog.text
