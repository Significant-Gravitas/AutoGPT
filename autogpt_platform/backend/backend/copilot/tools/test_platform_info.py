"""Tests for the get_platform_info tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.rate_limit import SubscriptionTier
from backend.copilot.tools.models import ResponseType
from backend.copilot.tools.platform_info import PlatformInfoTool

# Shorthand patch targets
_TIER_PATH = "backend.copilot.tools.platform_info.get_user_tier"
_FLAG_PATH = "backend.util.feature_flag.is_feature_enabled"


@pytest.fixture
def tool():
    return PlatformInfoTool()


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.session_id = "test-session-123"
    return session


def _billing_enabled():
    """Mock ``is_feature_enabled`` returning True (billing ON)."""
    return patch(_FLAG_PATH, new_callable=AsyncMock, return_value=True)


def _billing_disabled():
    """Mock ``is_feature_enabled`` returning False (billing OFF)."""
    return patch(_FLAG_PATH, new_callable=AsyncMock, return_value=False)


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
        with (
            _billing_enabled(),
            patch(
                _TIER_PATH, new_callable=AsyncMock, return_value=SubscriptionTier.PRO
            ),
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.PLATFORM_INFO
        assert result.tier == "PRO"
        assert result.billing_url == "/settings/billing"
        assert "Pro" in result.message
        assert "Settings" in result.message

    @pytest.mark.asyncio
    async def test_subscription_topic_no_tier(self, tool, mock_session):
        with (
            _billing_enabled(),
            patch(
                _TIER_PATH,
                new_callable=AsyncMock,
                return_value=SubscriptionTier.NO_TIER,
            ),
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.PLATFORM_INFO
        assert result.tier == "NO_TIER"
        assert "No active subscription" in result.message

    @pytest.mark.asyncio
    async def test_subscription_topic_max(self, tool, mock_session):
        with (
            _billing_enabled(),
            patch(
                _TIER_PATH, new_callable=AsyncMock, return_value=SubscriptionTier.MAX
            ),
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.tier == "MAX"
        assert "Max" in result.message

    @pytest.mark.asyncio
    async def test_subscription_topic_business(self, tool, mock_session):
        """Business tier is valid — user can be on it, we just don't advertise it."""
        with (
            _billing_enabled(),
            patch(
                _TIER_PATH,
                new_callable=AsyncMock,
                return_value=SubscriptionTier.BUSINESS,
            ),
        ):
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.tier == "BUSINESS"
        assert "Business" in result.message

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
        with (
            _billing_enabled(),
            patch(
                _TIER_PATH,
                new_callable=AsyncMock,
                side_effect=Exception("DB down"),
            ),
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

    @pytest.mark.asyncio
    async def test_response_mentions_autogpt_platform(self, tool, mock_session):
        """Platform identity context is in the response, not the description."""
        with (
            _billing_enabled(),
            patch(_TIER_PATH, return_value=SubscriptionTier.PRO),
        ):
            result = await tool._execute(
                user_id="u1", session=mock_session, topic="subscription"
            )
        assert "AutoGPT" in result.message
        assert "AutoPilot" in result.message

    # -- Feature flag: billing disabled (self-hosted / beta) --

    @pytest.mark.asyncio
    async def test_billing_disabled_returns_open_access(self, tool, mock_session):
        """When ENABLE_PLATFORM_PAYMENT is off, report open access."""
        with _billing_disabled():
            result = await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        assert result.type == ResponseType.PLATFORM_INFO
        assert result.tier == "OPEN_ACCESS"
        assert result.billing_url is None
        assert "open access" in result.message.lower()
        assert "AutoPilot" in result.message

    @pytest.mark.asyncio
    async def test_billing_disabled_skips_tier_lookup(self, tool, mock_session):
        """When billing is off, get_user_tier should never be called."""
        tier_mock = AsyncMock()
        with (
            _billing_disabled(),
            patch(_TIER_PATH, tier_mock),
        ):
            await tool._execute(
                user_id="user-1", session=mock_session, topic="subscription"
            )

        tier_mock.assert_not_called()
