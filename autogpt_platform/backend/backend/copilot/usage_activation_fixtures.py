"""Explicit billing-state boundary for unrelated provider/queue unit tests."""

from unittest.mock import AsyncMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data.pro_activation import UsageActivationState


@pytest.fixture(autouse=True)
def usage_snapshot(mocker):
    lookup = AsyncMock(
        side_effect=lambda user_id: UsageActivationState(
            user_id=user_id, tier=SubscriptionTier.NO_TIER, ready=True
        )
    )
    mocker.patch("backend.copilot.trial_cost_context.get_ready_usage_state", lookup)
    return lookup
