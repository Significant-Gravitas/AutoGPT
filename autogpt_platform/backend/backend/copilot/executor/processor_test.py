"""Unit tests for CoPilot mode routing logic in the processor.

Tests cover the mode→service mapping:
  - 'fast' → baseline service
  - 'extended_thinking' → SDK service
  - None → feature flag / config fallback

as well as the ``CHAT_MODE_OPTION`` server-side gate.  The tests import
the real production helpers from ``processor.py`` so the routing logic
has meaningful coverage.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.executor.processor import (
    resolve_effective_mode,
    resolve_use_sdk_for_mode,
)


class TestResolveUseSdkForMode:
    """Tests for the per-request mode routing logic."""

    @pytest.mark.asyncio
    async def test_fast_mode_uses_baseline(self):
        """mode='fast' always routes to baseline, regardless of flags."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    "fast",
                    "user-1",
                    use_claude_code_subscription=True,
                    config_default=True,
                )
                is False
            )

    @pytest.mark.asyncio
    async def test_extended_thinking_uses_sdk(self):
        """mode='extended_thinking' always routes to SDK, regardless of flags."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    "extended_thinking",
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_uses_subscription_override(self):
        """mode=None with claude_code_subscription=True routes to SDK."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=True,
                    config_default=False,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_uses_feature_flag(self):
        """mode=None with feature flag enabled routes to SDK."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ) as flag_mock:
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is True
            )
            flag_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_mode_uses_config_default(self):
        """mode=None falls back to config.use_claude_agent_sdk."""
        # When LaunchDarkly returns the default (True), we expect SDK routing.
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=True,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_all_disabled(self):
        """mode=None with all flags off routes to baseline."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is False
            )


class TestResolveEffectiveMode:
    """Tests for the CHAT_MODE_OPTION server-side gate."""

    @pytest.mark.asyncio
    async def test_none_mode_passes_through(self):
        """mode=None is returned as-is without a flag check."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ) as flag_mock:
            assert await resolve_effective_mode(None, "user-1") is None
            flag_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mode_stripped_when_flag_disabled(self):
        """When CHAT_MODE_OPTION is off, mode is dropped to None."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert await resolve_effective_mode("fast", "user-1") is None
            assert await resolve_effective_mode("extended_thinking", "user-1") is None

    @pytest.mark.asyncio
    async def test_mode_preserved_when_flag_enabled(self):
        """When CHAT_MODE_OPTION is on, the user-selected mode is preserved."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert await resolve_effective_mode("fast", "user-1") == "fast"
            assert (
                await resolve_effective_mode("extended_thinking", "user-1")
                == "extended_thinking"
            )

    @pytest.mark.asyncio
    async def test_anonymous_user_with_mode(self):
        """Anonymous users (user_id=None) still pass through the gate."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ) as flag_mock:
            assert await resolve_effective_mode("fast", None) is None
            flag_mock.assert_awaited_once()
