"""Tests for the get_platform_info tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.rate_limit import SubscriptionTier
from backend.copilot.tools.models import ResponseType
from backend.copilot.tools.platform_info import PlatformInfoTool


@pytest.fixture
def tool():
    return PlatformInfoTool()


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.session_id = "test-session-123"
    return session


class TestPlatformInfoTool:
    def test_name(self, tool):
        assert tool.name == "get_platform_info"

    def test_requires_auth(self, tool):
        assert tool.requires_auth is True

    def test_is_available(self, tool):
        assert tool.is_available is True

    def test_parameters_schema(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "topic" in params["properties"]
        assert "subscription" in params["properties"]["topic"]["enum"]

    @pytest.mark.asyncio
    async def test_subscription_topic_pro(self, tool, mock_session):
        with patch(
            "backend.copilot.tools.platform_info.get_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.PRO,
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.PLATFORM_INFO
        assert result.tier == "PRO"
        assert result.tier_multiplier == 5.0
        assert result.workspace_storage_mb == 1024
        assert result.billing_url == "/settings/billing"
        assert "upgrade_options" in result.data

    @pytest.mark.asyncio
    async def test_subscription_topic_no_tier(self, tool, mock_session):
        with patch(
            "backend.copilot.tools.platform_info.get_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.NO_TIER,
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.PLATFORM_INFO
        assert result.tier == "NO_TIER"
        assert result.tier_multiplier == 0.0
        # Should show upgrade options for tiers above NO_TIER
        assert len(result.data["upgrade_options"]) > 0

    @pytest.mark.asyncio
    async def test_subscription_topic_max(self, tool, mock_session):
        with patch(
            "backend.copilot.tools.platform_info.get_user_tier",
            new_callable=AsyncMock,
            return_value=SubscriptionTier.MAX,
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.tier == "MAX"
        assert result.tier_multiplier == 20.0
        assert result.workspace_storage_mb == 5 * 1024
        # Only Business should be in upgrade options (Enterprise excluded)
        upgrade_tiers = [o["tier"] for o in result.data["upgrade_options"]]
        assert "BUSINESS" in upgrade_tiers
        assert "ENTERPRISE" not in upgrade_tiers

    @pytest.mark.asyncio
    async def test_subscription_no_user_id(self, tool, mock_session):
        result = await tool._execute(
            user_id=None, session=mock_session, topic="subscription"
        )
        assert result.type == ResponseType.ERROR

    @pytest.mark.asyncio
    async def test_invalid_topic(self, tool, mock_session):
        result = await tool._execute(
            user_id="user-1", session=mock_session, topic="invalid"
        )
        assert result.type == ResponseType.ERROR
        assert "Unknown topic" in result.message

    @pytest.mark.asyncio
    async def test_tier_lookup_failure(self, tool, mock_session):
        with patch(
            "backend.copilot.tools.platform_info.get_user_tier",
            new_callable=AsyncMock,
            side_effect=Exception("DB down"),
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.ERROR
        assert "Could not retrieve" in result.message

    def test_tool_appears_in_registry(self):
        from backend.copilot.tools import TOOL_REGISTRY

        assert "get_platform_info" in TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["get_platform_info"], PlatformInfoTool)

    def test_openai_schema(self, tool):
        schema = tool.as_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_platform_info"
