from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.platform_cost import get_copilot_weekly_usage_for_export


@pytest.mark.asyncio
async def test_trial_export_uses_frozen_weekly_limit():
    with (
        patch(
            "backend.data.platform_cost.query_raw_with_schema",
            AsyncMock(
                return_value=[
                    {
                        "user_id": "trial-user",
                        "user_email": "trial@example.com",
                        "tier": "TRIAL",
                        "week_start": "2026-09-07T00:00:00Z",
                        "cost_microdollars": 500_000,
                        "trial_weekly_limit": 1_000_000,
                    }
                ]
            ),
        ) as query,
        patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            AsyncMock(return_value={"TRIAL": 0}),
        ),
    ):
        rows = await get_copilot_weekly_usage_for_export(
            datetime(2026, 9, 7, tzinfo=UTC), datetime(2026, 9, 8, tzinfo=UTC)
        )
    assert rows[0].weekly_limit_microdollars == 1_000_000
    assert rows[0].percent_used == 50
    assert "SubscriptionTrial" in query.call_args.args[0]


@pytest.mark.asyncio
async def test_trial_export_rejects_missing_accepted_limit():
    with (
        patch(
            "backend.data.platform_cost.query_raw_with_schema",
            AsyncMock(
                return_value=[
                    {
                        "user_id": "trial-user",
                        "tier": "TRIAL",
                        "week_start": "2026-09-07T00:00:00Z",
                        "cost_microdollars": 500_000,
                        "trial_weekly_limit": None,
                    }
                ]
            ),
        ),
        patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            AsyncMock(return_value={"TRIAL": 0}),
        ),
    ):
        with pytest.raises(ValueError, match="accepted weekly limit"):
            await get_copilot_weekly_usage_for_export(
                datetime(2026, 9, 7, tzinfo=UTC), datetime(2026, 9, 8, tzinfo=UTC)
            )
