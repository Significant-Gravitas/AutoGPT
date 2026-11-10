from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import (
    RateLimitUnavailable,
    SubscriptionTier,
    get_usage_status,
)
from backend.copilot.usage_activation import usage_keys, verify_activation_usage
from backend.data.pro_activation import UsageActivationState


def test_generation_preserves_calendar_boundaries():
    now = datetime(2027, 1, 1, 23, 59, tzinfo=UTC)
    daily, weekly = usage_keys("user", "activation", now)
    assert daily == "copilot:cost:daily:user:activation:2027-01-01"
    assert weekly == "copilot:cost:weekly:user:activation:2026-W53"


@pytest.mark.asyncio
async def test_activation_verification_never_overwrites_paid_usage():
    redis = AsyncMock()
    redis.get.side_effect = [b"123", b"456", b"123", b"456"]
    with patch("backend.copilot.usage_activation.get_redis_async", return_value=redis):
        await verify_activation_usage("user", "activation")
        await verify_activation_usage("user", "activation")
    redis.set.assert_not_called()
    redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_paid_usage_gauge_never_returns_false_zero_during_redis_failure():
    state = UsageActivationState(
        user_id="user", generation="activation", tier="PRO", ready=True
    )
    with (
        patch("backend.copilot.rate_limit.get_ready_usage_state", return_value=state),
        patch(
            "backend.copilot.rate_limit.get_redis_async",
            side_effect=ConnectionError("Redis temporarily unavailable"),
        ),
    ):
        with pytest.raises(RateLimitUnavailable):
            await get_usage_status("user", 100, 500, tier=SubscriptionTier.PRO)
