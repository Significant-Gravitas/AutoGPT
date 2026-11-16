"""The usage caps an MCP tool call checks before it spends."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from mcp.server.fastmcp.exceptions import ToolError

from backend.api.external.v2 import mcp_calls
from backend.copilot.limit_tracking import LimitReading
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable
from backend.util import posthog_client


@pytest.fixture
def limits(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    mocker.patch.object(
        mcp_calls,
        "get_global_rate_limits",
        AsyncMock(return_value=(1_000_000, 5_000_000, "PRO")),
    )
    check = AsyncMock()
    mocker.patch.object(mcp_calls, "check_rate_limit", check)
    return check


@pytest.mark.asyncio
async def test_a_tool_call_over_the_cap_is_refused_and_sent(
    limits: AsyncMock, chat_blocked_events: list[dict]
):
    resets_at = datetime.now(UTC) + timedelta(days=2)
    limits.side_effect = RateLimitExceeded(
        "weekly", resets_at, reading=LimitReading(tier="PRO", used=5.0, cap=5.0)
    )

    with pytest.raises(ToolError, match="weekly usage limit"):
        await mcp_calls.check_spend_allowance("user-1")

    assert chat_blocked_events == [
        {
            "distinct_id": "user-1",
            "reason": "usage_limit",
            "limit": "weekly",
            "outcome": "refused",
            "surface": "api",
            "tier": "PRO",
            "used": 5.0,
            "cap": 5.0,
            "resets_at": resets_at.isoformat(),
        }
    ]


@pytest.mark.asyncio
async def test_an_unreadable_cap_refuses_without_an_event(
    limits: AsyncMock, chat_blocked_events: list[dict]
):
    limits.side_effect = RateLimitUnavailable()

    with pytest.raises(ToolError, match="unavailable"):
        await mcp_calls.check_spend_allowance("user-1")

    assert chat_blocked_events == []


@pytest.mark.asyncio
async def test_a_tool_call_over_the_cap_is_refused_the_same_when_posthog_fails(
    limits: AsyncMock, monkeypatch: pytest.MonkeyPatch
):
    def broken(*_args, **_kwargs):
        raise RuntimeError("posthog down")

    monkeypatch.setattr(posthog_client, "capture", broken)
    limits.side_effect = RateLimitExceeded(
        "daily", datetime.now(UTC) + timedelta(hours=3)
    )

    with pytest.raises(ToolError, match="daily usage limit"):
        await mcp_calls.check_spend_allowance("user-1")
