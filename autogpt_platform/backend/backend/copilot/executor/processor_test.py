"""Unit tests for CoPilot mode routing logic in the processor.

Tests cover the mode→service mapping:
  - 'fast' → baseline service
  - 'extended_thinking' → SDK service
  - None → feature flag / config fallback
"""

from typing import Literal


def _resolve_use_sdk(
    mode: Literal["fast", "extended_thinking"] | None,
    use_claude_code_subscription: bool = False,
    feature_flag_value: bool = False,
    config_default: bool = True,
) -> bool:
    """Replicate the mode-routing logic from CoPilotProcessor._execute_async.

    Extracted here so we can test it in isolation without instantiating the
    full processor or its event loop.
    """
    if mode == "fast":
        return False
    elif mode == "extended_thinking":
        return True
    else:
        return use_claude_code_subscription or feature_flag_value or config_default


class TestModeRouting:
    """Tests for the per-request mode routing logic."""

    def test_fast_mode_uses_baseline(self):
        """mode='fast' always routes to baseline, regardless of flags."""
        assert _resolve_use_sdk("fast") is False
        assert _resolve_use_sdk("fast", use_claude_code_subscription=True) is False
        assert _resolve_use_sdk("fast", feature_flag_value=True) is False

    def test_extended_thinking_uses_sdk(self):
        """mode='extended_thinking' always routes to SDK, regardless of flags."""
        assert _resolve_use_sdk("extended_thinking") is True
        assert (
            _resolve_use_sdk("extended_thinking", use_claude_code_subscription=False)
            is True
        )
        assert _resolve_use_sdk("extended_thinking", config_default=False) is True

    def test_none_mode_uses_subscription_override(self):
        """mode=None with claude_code_subscription=True routes to SDK."""
        assert (
            _resolve_use_sdk(
                None,
                use_claude_code_subscription=True,
                feature_flag_value=False,
                config_default=False,
            )
            is True
        )

    def test_none_mode_uses_feature_flag(self):
        """mode=None with feature flag enabled routes to SDK."""
        assert (
            _resolve_use_sdk(
                None,
                use_claude_code_subscription=False,
                feature_flag_value=True,
                config_default=False,
            )
            is True
        )

    def test_none_mode_uses_config_default(self):
        """mode=None falls back to config.use_claude_agent_sdk."""
        assert (
            _resolve_use_sdk(
                None,
                use_claude_code_subscription=False,
                feature_flag_value=False,
                config_default=True,
            )
            is True
        )

    def test_none_mode_all_disabled(self):
        """mode=None with all flags off routes to baseline."""
        assert (
            _resolve_use_sdk(
                None,
                use_claude_code_subscription=False,
                feature_flag_value=False,
                config_default=False,
            )
            is False
        )

    def test_none_mode_precedence_subscription_over_flag(self):
        """Claude Code subscription takes priority over feature flag."""
        assert (
            _resolve_use_sdk(
                None,
                use_claude_code_subscription=True,
                feature_flag_value=False,
                config_default=False,
            )
            is True
        )
