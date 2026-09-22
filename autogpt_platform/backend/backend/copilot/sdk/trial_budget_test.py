from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.sdk.service import _resolve_dynamic_max_budget_usd


@pytest.fixture
def trial_budget():
    with (
        patch(
            "backend.copilot.sdk.service.get_global_rate_limits",
            AsyncMock(return_value=(250_000, 1_000_000, "TRIAL")),
        ),
        patch(
            "backend.copilot.sdk.service.get_remaining_usd_budget", AsyncMock()
        ) as remaining,
        patch("backend.copilot.sdk.service.config.claude_agent_max_budget_usd", 10.0),
    ):
        yield remaining


@pytest.mark.asyncio
async def test_trial_budget_is_never_rounded_up_to_paid_floor(trial_budget):
    trial_budget.return_value = 0.05
    assert await _resolve_dynamic_max_budget_usd("trial-user") == 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan")])
async def test_unavailable_trial_budget_refuses_sdk_dispatch(trial_budget, remaining):
    trial_budget.return_value = remaining
    with pytest.raises(ValueError, match="Trial budget"):
        await _resolve_dynamic_max_budget_usd("trial-user")


@pytest.mark.asyncio
async def test_trial_still_respects_static_per_turn_cap(trial_budget):
    trial_budget.return_value = 20.0
    assert await _resolve_dynamic_max_budget_usd("trial-user") == 10.0
