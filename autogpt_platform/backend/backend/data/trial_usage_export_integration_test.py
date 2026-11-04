from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import PlatformCostLog, User

from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.platform_cost import get_copilot_weekly_usage_for_export

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


@pytest.mark.asyncio
async def test_export_joins_accepted_trial_terms_without_multiplying_usage(enrollment):
    start = datetime(2026, 9, 7, tzinfo=UTC)
    await User.prisma().update(
        where={"id": enrollment.user_id},
        data={"subscriptionTier": SubscriptionTier.TRIAL},
    )
    try:
        for cost in (15, 25):
            await PlatformCostLog.prisma().create(
                data={
                    "userId": enrollment.user_id,
                    "createdAt": start,
                    "provider": "test",
                    "blockName": "copilot:test",
                    "costMicrodollars": cost,
                }
            )
        with patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            AsyncMock(return_value={"TRIAL": 0}),
        ):
            rows = await get_copilot_weekly_usage_for_export(
                start, start + timedelta(days=1)
            )
        (row,) = [row for row in rows if row.user_id == enrollment.user_id]
        assert row.copilot_cost_microdollars == 40
        assert (
            row.weekly_limit_microdollars == enrollment.offer.weekly_cost_limit == 100
        )
        assert row.percent_used == 40
    finally:
        await PlatformCostLog.prisma().delete_many(where={"userId": enrollment.user_id})
