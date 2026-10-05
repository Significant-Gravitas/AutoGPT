"""The pre-turn credential check reads the key each SDK route uses."""

from unittest.mock import patch

import pytest

from backend.copilot import config as cfg_mod

from .service import _deployment_route_has_credential


class TestDeploymentRouteHasCredential:
    """The pre-turn credential check reads the key each route actually uses."""

    @pytest.mark.parametrize(
        "subscription, openrouter_active, direct_key, expected",
        [
            (True, False, None, True),
            (False, True, None, True),
            (False, False, "sk-ant-key", True),
            (False, False, None, False),
        ],
    )
    def test_route_credentials(
        self, subscription, openrouter_active, direct_key, expected
    ) -> None:
        with patch("backend.copilot.sdk.service.config") as mock_cfg:
            mock_cfg.use_claude_code_subscription = subscription
            mock_cfg.openrouter_active = openrouter_active
            mock_cfg.direct_anthropic_api_key = direct_key
            assert _deployment_route_has_credential() is expected

    def test_direct_route_with_only_anthropic_api_key(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ) -> None:
        """``ANTHROPIC_API_KEY`` alone selects direct Anthropic and is enough
        to run it, though ``api_key`` never reads that variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-only-key")
        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        assert cfg.effective_transport == "direct_anthropic"
        assert not cfg.api_key
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _deployment_route_has_credential() is True
