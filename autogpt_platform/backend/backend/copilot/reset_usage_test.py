"""Unit tests for the POST /usage/reset endpoint."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow
from backend.util.exceptions import InsufficientBalanceError


# Minimal config mock matching ChatConfig fields used by the endpoint.
def _make_config(
    rate_limit_reset_cost: int = 200,
    daily_token_limit: int = 2_500_000,
    weekly_token_limit: int = 12_500_000,
    max_daily_resets: int = 5,
):
    cfg = MagicMock()
    cfg.rate_limit_reset_cost = rate_limit_reset_cost
    cfg.daily_token_limit = daily_token_limit
    cfg.weekly_token_limit = weekly_token_limit
    cfg.max_daily_resets = max_daily_resets
    return cfg


def _usage(daily_used: int = 3_000_000, daily_limit: int = 2_500_000):
    return CoPilotUsageStatus(
        daily=UsageWindow(
            used=daily_used,
            limit=daily_limit,
            resets_at=datetime.now(UTC) + timedelta(hours=6),
        ),
        weekly=UsageWindow(
            used=5_000_000,
            limit=12_500_000,
            resets_at=datetime.now(UTC) + timedelta(days=3),
        ),
    )


_MODULE = "backend.api.features.chat.routes"


@pytest.mark.asyncio
class TestResetCopilotUsage:
    async def test_feature_disabled_returns_400(self):
        """When rate_limit_reset_cost=0, endpoint returns 400."""
        from backend.api.features.chat.routes import reset_copilot_usage

        with patch(f"{_MODULE}.config", _make_config(rate_limit_reset_cost=0)):
            with pytest.raises(HTTPException) as exc_info:
                await reset_copilot_usage(user_id="user-1")
            assert exc_info.value.status_code == 400
            assert "not available" in exc_info.value.detail

    async def test_not_at_limit_returns_400(self):
        """When user hasn't hit their daily limit, returns 400."""
        from backend.api.features.chat.routes import reset_copilot_usage

        cfg = _make_config()
        with (
            patch(f"{_MODULE}.config", cfg),
            patch(f"{_MODULE}.get_daily_reset_count", AsyncMock(return_value=0)),
            patch(f"{_MODULE}.acquire_reset_lock", AsyncMock(return_value=True)),
            patch(f"{_MODULE}.release_reset_lock", AsyncMock()),
            patch(
                f"{_MODULE}.get_usage_status",
                AsyncMock(return_value=_usage(daily_used=1_000_000)),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reset_copilot_usage(user_id="user-1")
            assert exc_info.value.status_code == 400
            assert "not reached" in exc_info.value.detail

    async def test_insufficient_credits_returns_402(self):
        """When user doesn't have enough credits, returns 402."""
        from backend.api.features.chat.routes import reset_copilot_usage

        mock_credit_model = AsyncMock()
        mock_credit_model.spend_credits.side_effect = InsufficientBalanceError(
            message="Insufficient balance",
            user_id="user-1",
            balance=50,
            amount=200,
        )

        cfg = _make_config()
        with (
            patch(f"{_MODULE}.config", cfg),
            patch(f"{_MODULE}.get_daily_reset_count", AsyncMock(return_value=0)),
            patch(f"{_MODULE}.acquire_reset_lock", AsyncMock(return_value=True)),
            patch(f"{_MODULE}.release_reset_lock", AsyncMock()),
            patch(
                f"{_MODULE}.get_usage_status",
                AsyncMock(return_value=_usage()),
            ),
            patch(
                f"{_MODULE}.get_user_credit_model",
                AsyncMock(return_value=mock_credit_model),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reset_copilot_usage(user_id="user-1")
            assert exc_info.value.status_code == 402

    async def test_happy_path(self):
        """Successful reset: charges credits, resets usage, returns response."""
        from backend.api.features.chat.routes import reset_copilot_usage

        mock_credit_model = AsyncMock()
        mock_credit_model.spend_credits.return_value = 1800  # remaining balance

        cfg = _make_config()
        updated_usage = _usage(daily_used=0)

        with (
            patch(f"{_MODULE}.config", cfg),
            patch(f"{_MODULE}.get_daily_reset_count", AsyncMock(return_value=0)),
            patch(f"{_MODULE}.acquire_reset_lock", AsyncMock(return_value=True)),
            patch(f"{_MODULE}.release_reset_lock", AsyncMock()),
            patch(
                f"{_MODULE}.get_usage_status",
                AsyncMock(side_effect=[_usage(), updated_usage]),
            ),
            patch(
                f"{_MODULE}.get_user_credit_model",
                AsyncMock(return_value=mock_credit_model),
            ),
            patch(f"{_MODULE}.reset_daily_usage", AsyncMock(return_value=True)),
            patch(f"{_MODULE}.increment_daily_reset_count", AsyncMock()),
        ):
            result = await reset_copilot_usage(user_id="user-1")
            assert result.success is True
            assert result.credits_charged == 200
            assert result.remaining_balance == 1800

    async def test_max_daily_resets_exceeded(self):
        """When user has exhausted daily resets, returns 429."""
        from backend.api.features.chat.routes import reset_copilot_usage

        cfg = _make_config(max_daily_resets=3)
        with (
            patch(f"{_MODULE}.config", cfg),
            patch(f"{_MODULE}.get_daily_reset_count", AsyncMock(return_value=3)),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reset_copilot_usage(user_id="user-1")
            assert exc_info.value.status_code == 429
