import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.util.subscription_tier_order import (
    SUBSCRIPTION_TIER_ORDER,
    subscription_tier_at_least,
    subscription_tier_rank,
)
from backend.util.subscription_tiers import (
    SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS,
    SubscriptionTierCacheInvalidationError,
    _generation_key,
    invalidate_subscription_tier_auxiliary_caches,
    invalidate_subscription_tier_caches,
)


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
