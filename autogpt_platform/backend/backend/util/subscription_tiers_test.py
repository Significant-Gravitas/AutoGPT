import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.util import cache as cache_module
from backend.util import subscription_tiers
from backend.util.subscription_tier_order import (
    SUBSCRIPTION_TIER_ORDER,
    subscription_tier_at_least,
    subscription_tier_rank,
)
from backend.util.subscription_tiers import (
    SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS,
    SubscriptionTierCacheInvalidationError,
    SubscriptionTierUserNotFoundError,
    _generation_key,
    invalidate_subscription_tier_auxiliary_caches,
    invalidate_subscription_tier_caches,
)
from backend.util.service import UnhealthyServiceError


class _MemoryValueRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def setex(self, key: str, _ttl: int, value: bytes) -> bool:
        self.values[key] = value
        return True


class _MemoryGenerationRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.fail_read = False

    async def eval(
        self,
        script: str,
        _num_keys: int,
        key: str,
        value: str,
        _ttl: int,
    ) -> str:
        if self.fail_read:
            raise ConnectionError("Redis unavailable")
        assert 'redis.call("GET"' in script
        self.values.setdefault(key, value)
        return self.values[key]

    async def set(self, key: str, value: str, *, ex: int) -> bool:
        assert ex == SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS
        self.values[key] = value
        return True


@pytest.fixture
def isolated_tier_cache(monkeypatch: pytest.MonkeyPatch):
    values = _MemoryValueRedis()
    generations = _MemoryGenerationRedis()

    async def get_generation_redis() -> _MemoryGenerationRedis:
        return generations

    monkeypatch.setattr(cache_module, "_get_redis", lambda: values)
    monkeypatch.setattr(
        subscription_tiers,
        "get_redis_async",
        get_generation_redis,
    )
    return values, generations


def _database_client(*tiers: SubscriptionTier | Exception):
    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=tiers)
    return client


def test_subscription_tier_order_covers_every_enum_value_once():
    assert len(SUBSCRIPTION_TIER_ORDER) == len(set(SUBSCRIPTION_TIER_ORDER))
    assert set(SUBSCRIPTION_TIER_ORDER) == set(SubscriptionTier)
    assert [subscription_tier_rank(tier) for tier in SUBSCRIPTION_TIER_ORDER] == list(
        range(len(SUBSCRIPTION_TIER_ORDER))
    )


def test_subscription_tier_at_least_uses_canonical_order():
    assert subscription_tier_at_least(SubscriptionTier.MAX, SubscriptionTier.MAX)
    assert subscription_tier_at_least(SubscriptionTier.ENTERPRISE, SubscriptionTier.MAX)
    assert not subscription_tier_at_least(SubscriptionTier.PRO, SubscriptionTier.MAX)


@pytest.mark.asyncio
async def test_dual_invalidation_rotates_generation_and_deletes_legacy_cache():
    redis = MagicMock()
    redis.set = AsyncMock(return_value=True)
    legacy_delete = MagicMock(return_value=False)

    with patch(
        "backend.util.subscription_tiers.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await invalidate_subscription_tier_caches("user-1", legacy_delete)

    redis.set.assert_awaited_once()
    args, kwargs = redis.set.await_args
    assert args[0] == _generation_key("user-1")
    assert isinstance(args[1], str) and len(args[1]) == 32
    assert kwargs == {"ex": SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS}
    legacy_delete.assert_called_once_with("user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_error", "legacy_error", "failed_caches"),
    [
        (ConnectionError("generation down"), None, ("generation",)),
        (None, ConnectionError("legacy down"), ("legacy",)),
        (
            ConnectionError("generation down"),
            ConnectionError("legacy down"),
            ("generation", "legacy"),
        ),
    ],
)
async def test_dual_invalidation_attempts_both_legs_before_raising(
    generation_error: Exception | None,
    legacy_error: Exception | None,
    failed_caches: tuple[str, ...],
):
    events = MagicMock()
    redis = MagicMock()

    async def set_generation(*_args, **_kwargs):
        events("generation")
        if generation_error:
            raise generation_error
        return True

    def delete_legacy(_user_id: str):
        events("legacy")
        if legacy_error:
            raise legacy_error
        return False

    redis.set = AsyncMock(side_effect=set_generation)
    with patch(
        "backend.util.subscription_tiers.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        with pytest.raises(SubscriptionTierCacheInvalidationError) as exc_info:
            await invalidate_subscription_tier_caches("user-1", delete_legacy)

    assert exc_info.value.failed_caches == failed_caches
    assert events.call_args_list == [call("generation"), call("legacy")]


@pytest.mark.asyncio
async def test_unacknowledged_generation_write_is_a_failure_but_legacy_is_attempted():
    redis = MagicMock()
    redis.set = AsyncMock(return_value=False)
    legacy_delete = MagicMock(return_value=False)

    with patch(
        "backend.util.subscription_tiers.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        with pytest.raises(SubscriptionTierCacheInvalidationError) as exc_info:
            await invalidate_subscription_tier_caches("user-1", legacy_delete)

    assert exc_info.value.failed_caches == ("generation",)
    legacy_delete.assert_called_once_with("user-1")


def test_auxiliary_invalidation_attempts_every_cache_after_failure():
    first_delete = MagicMock(side_effect=ConnectionError("first cache down"))
    second_delete = MagicMock()
    credit_module = MagicMock()
    credit_module.get_user_by_id.cache_delete = first_delete
    credit_module.get_pending_subscription_change.cache_delete = second_delete

    with patch.dict("sys.modules", {"backend.data.credit": credit_module}):
        invalidate_subscription_tier_auxiliary_caches("user-1")

    first_delete.assert_called_once_with("user-1")
    second_delete.assert_called_once_with("user-1")


@pytest.mark.asyncio
async def test_cancellation_waits_for_both_invalidations_then_propagates():
    generation_started = asyncio.Event()
    finish_generation = asyncio.Event()
    events: list[str] = []

    async def rotate(_user_id: str) -> None:
        events.append("generation-started")
        generation_started.set()
        await finish_generation.wait()
        events.append("generation-finished")

    def delete_legacy(_user_id: str) -> None:
        events.append("legacy")

    with patch(
        "backend.util.subscription_tiers._rotate_subscription_tier_generation",
        side_effect=rotate,
    ):
        caller = asyncio.create_task(
            invalidate_subscription_tier_caches("user-1", delete_legacy)
        )
        await generation_started.wait()
        caller.cancel()
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.sleep(0)

        assert not caller.done()
        finish_generation.set()

        with pytest.raises(asyncio.CancelledError):
            await caller

    assert events == ["generation-started", "generation-finished", "legacy"]


@pytest.mark.asyncio
async def test_cancellation_wins_when_invalidation_also_fails():
    generation_started = asyncio.Event()
    fail_generation = asyncio.Event()
    legacy_delete = MagicMock()

    async def rotate(_user_id: str) -> None:
        generation_started.set()
        await fail_generation.wait()
        raise ConnectionError("generation down")

    with patch(
        "backend.util.subscription_tiers._rotate_subscription_tier_generation",
        side_effect=rotate,
    ):
        caller = asyncio.create_task(
            invalidate_subscription_tier_caches("user-1", legacy_delete)
        )
        await generation_started.wait()
        caller.cancel()
        fail_generation.set()

        with pytest.raises(asyncio.CancelledError):
            await caller

    legacy_delete.assert_called_once_with("user-1")


def test_generation_key_hides_user_id_and_is_user_specific():
    first = _generation_key("person@example.com")
    second = _generation_key("someone-else@example.com")

    assert "person@example.com" not in first
    assert first == _generation_key("person@example.com")
    assert first != second


@pytest.mark.asyncio
async def test_successful_lookup_is_shared_across_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(SubscriptionTier.MAX)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    results = await asyncio.gather(
        *(
            subscription_tiers.get_authoritative_subscription_tier("same-user")
            for _ in range(8)
        )
    )

    assert results == [SubscriptionTier.MAX] * 8
    client.get_user_subscription_tier.assert_awaited_once_with("same-user")


@pytest.mark.asyncio
async def test_different_users_are_cached_without_cross_user_leakage(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    async def read_tier(user_id: str) -> SubscriptionTier:
        return {
            "max-user": SubscriptionTier.MAX,
            "pro-user": SubscriptionTier.PRO,
        }[user_id]

    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=read_tier)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    first = await asyncio.gather(
        subscription_tiers.get_authoritative_subscription_tier("max-user"),
        subscription_tiers.get_authoritative_subscription_tier("pro-user"),
    )
    second = await asyncio.gather(
        subscription_tiers.get_authoritative_subscription_tier("max-user"),
        subscription_tiers.get_authoritative_subscription_tier("pro-user"),
    )

    assert first == second == [SubscriptionTier.MAX, SubscriptionTier.PRO]
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_missing_user_and_transient_failure_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(
        ValueError("missing"),
        RuntimeError("database unavailable"),
        SubscriptionTier.BUSINESS,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    with pytest.raises(SubscriptionTierUserNotFoundError):
        await subscription_tiers.get_authoritative_subscription_tier("retry-user")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await subscription_tiers.get_authoritative_subscription_tier("retry-user")
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("retry-user")
        == SubscriptionTier.BUSINESS
    )
    assert client.get_user_subscription_tier.await_count == 3


@pytest.mark.asyncio
async def test_unhealthy_database_manager_is_not_mapped_to_missing_user(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(
        UnhealthyServiceError("database manager unavailable"),
        SubscriptionTier.MAX,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    with pytest.raises(UnhealthyServiceError, match="database manager unavailable"):
        await subscription_tiers.get_authoritative_subscription_tier("health-outage")
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("health-outage")
        == SubscriptionTier.MAX
    )
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_invalidation_is_selective(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
        SubscriptionTier.PRO,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("changed-user")
        == SubscriptionTier.MAX
    )
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("stable-user")
        == SubscriptionTier.BUSINESS
    )

    legacy_delete = MagicMock()
    await invalidate_subscription_tier_caches("changed-user", legacy_delete)

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("changed-user")
        == SubscriptionTier.PRO
    )
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("stable-user")
        == SubscriptionTier.BUSINESS
    )
    legacy_delete.assert_called_once_with("changed-user")
    assert client.get_user_subscription_tier.await_count == 3


@pytest.mark.asyncio
async def test_invalidation_fences_an_in_flight_stale_fill(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    first_read_started = asyncio.Event()
    release_first_read = asyncio.Event()
    calls = 0

    async def read_tier(_user_id: str) -> SubscriptionTier:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_read_started.set()
            await release_first_read.wait()
            return SubscriptionTier.MAX
        return SubscriptionTier.PRO

    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=read_tier)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    stale_read = asyncio.create_task(
        subscription_tiers.get_authoritative_subscription_tier("downgraded-user")
    )
    await first_read_started.wait()
    await invalidate_subscription_tier_caches("downgraded-user", MagicMock())
    release_first_read.set()

    assert await stale_read == SubscriptionTier.MAX
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("downgraded-user")
        == SubscriptionTier.PRO
    )
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_evicted_generation_never_reuses_an_old_cache_key(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(SubscriptionTier.MAX, SubscriptionTier.PRO)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("evicted-user")
        == SubscriptionTier.MAX
    )
    _, generations = isolated_tier_cache
    generations.values.clear()

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("evicted-user")
        == SubscriptionTier.PRO
    )
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_generation_read_failure_bypasses_value_cache(
    monkeypatch: pytest.MonkeyPatch,
    isolated_tier_cache,
):
    client = _database_client(SubscriptionTier.MAX, SubscriptionTier.PRO)
    monkeypatch.setattr(
        subscription_tiers,
        "get_database_manager_async_client",
        lambda: client,
    )
    values, generations = isolated_tier_cache
    generations.fail_read = True

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("redis-down")
        == SubscriptionTier.MAX
    )
    assert (
        await subscription_tiers.get_authoritative_subscription_tier("redis-down")
        == SubscriptionTier.PRO
    )
    assert values.values == {}
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_bytes_generation_is_decoded_before_value_lookup(
    monkeypatch: pytest.MonkeyPatch,
):
    redis = MagicMock()
    redis.eval = AsyncMock(return_value=b"byte-generation")

    async def get_generation_redis():
        return redis

    fetch_tier = AsyncMock(return_value=SubscriptionTier.MAX)
    monkeypatch.setattr(
        subscription_tiers,
        "get_redis_async",
        get_generation_redis,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "_fetch_subscription_tier_for_generation",
        fetch_tier,
    )

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("bytes-user")
        == SubscriptionTier.MAX
    )
    fetch_tier.assert_awaited_once_with("bytes-user", "byte-generation")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_generation", [None, "", b""])
async def test_invalid_generation_bypasses_value_cache(
    monkeypatch: pytest.MonkeyPatch,
    invalid_generation: str | bytes | None,
):
    redis = MagicMock()
    redis.eval = AsyncMock(return_value=invalid_generation)

    async def get_generation_redis():
        return redis

    load_tier = AsyncMock(return_value=SubscriptionTier.PRO)
    fetch_tier = AsyncMock()
    monkeypatch.setattr(
        subscription_tiers,
        "get_redis_async",
        get_generation_redis,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "_load_authoritative_subscription_tier",
        load_tier,
    )
    monkeypatch.setattr(
        subscription_tiers,
        "_fetch_subscription_tier_for_generation",
        fetch_tier,
    )

    assert (
        await subscription_tiers.get_authoritative_subscription_tier("invalid-user")
        == SubscriptionTier.PRO
    )
    load_tier.assert_awaited_once_with("invalid-user")
    fetch_tier.assert_not_awaited()
