"""Tests for the route guards in ``route_launch_env``: each route's
billing-header choice, credential and window from one profile, and the
orchestrator block's launch on a user's own key."""

from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import (
    codex_profile,
    platform_profile,
    user_key_profile,
)
from backend.copilot.sdk.env import GatewayEndpoint, route_launch_env

_ROUTES = ("openrouter", "direct", "subscription", "codex")
_GATEWAY = GatewayEndpoint(base_url="http://127.0.0.1:43210", auth_token="t")


def _env_for_route(route: str) -> dict[str, str]:
    """``route_launch_env`` for *route*, with no parent environment."""
    if route == "codex":
        profile = codex_profile(
            "gpt-5.6-terra", effort=None, routing_source="env", codex_engine=None
        )
    else:
        cfg = {
            "openrouter": ChatConfig(
                platform_route="openrouter", openrouter_api_key="sk-or-test"
            ),
            "direct": ChatConfig(
                platform_route="direct_anthropic",
                anthropic_api_key="sk-ant-direct-key",
            ),
            "subscription": ChatConfig(platform_route="subscription"),
        }[route]
        profile = platform_profile(cfg, "claude-sonnet-4-6", "env")
    with (
        patch("backend.copilot.sdk.env.validate_subscription"),
        patch.dict("backend.copilot.sdk.env.os.environ", {}, clear=True),
    ):
        return route_launch_env(profile, gateway=_GATEWAY if route == "codex" else None)


class TestAttributionHeaderPerRoute:
    """The CLI's billing-header system block stays where the CLI talks to
    Anthropic first-party and goes everywhere else: the route decides, no
    deployment switch."""

    @pytest.mark.parametrize(
        "route, expected",
        [
            ("openrouter", "0"),
            ("codex", "0"),
            ("direct", None),
            ("subscription", None),
        ],
    )
    def test_off_only_where_the_cli_is_not_first_party(self, route, expected):
        assert _env_for_route(route).get("CLAUDE_CODE_ATTRIBUTION_HEADER") == expected


class TestCredentialAndWindowComeFromOneRoute:
    """A launch's credential (``route_credentials``) and its window pin
    (``context_window``) follow the same route profile: no route takes one
    route's key with another route's window."""

    @pytest.mark.parametrize(
        "route, credential, base_url, window",
        [
            (
                "openrouter",
                ("ANTHROPIC_AUTH_TOKEN", "sk-or-test"),
                "https://openrouter.ai/api",
                "200000",
            ),
            ("direct", ("ANTHROPIC_API_KEY", "sk-ant-direct-key"), "", "1000000"),
            ("subscription", ("ANTHROPIC_API_KEY", ""), "", "200000"),
            (
                "codex",
                ("ANTHROPIC_AUTH_TOKEN", "t"),
                "http://127.0.0.1:43210",
                "272000",
            ),
        ],
    )
    def test_each_route_carries_its_credential_and_its_window(
        self, route, credential, base_url, window
    ):
        env = _env_for_route(route)

        key, value = credential
        assert env[key] == value
        assert env["ANTHROPIC_BASE_URL"] == base_url
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == window


# ---------------------------------------------------------------------------
# A launch on a user's own key: the orchestrator block's
# ---------------------------------------------------------------------------

_PARENT_WITH_OAUTH = {
    "ANTHROPIC_BASE_URL": "https://remote-gateway.example.com",
    "ANTHROPIC_API_KEY": "sk-ant-deployment-key",
    "ANTHROPIC_AUTH_TOKEN": "inherited-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "inherited-refresh",
}


def _user_key_env(provider: str, user_key: str) -> dict[str, str]:
    """The env the CLI sees: the parent's, under the launch's own."""
    with patch.dict("backend.copilot.sdk.env.os.environ", _PARENT_WITH_OAUTH, True):
        launch = route_launch_env(
            user_key_profile(provider, user_key, "claude-sonnet-4-6")
        )
    return {**_PARENT_WITH_OAUTH, **launch}


class TestUserKeyLaunch:
    """A launch on a user's own key carries that key and nothing else."""

    def test_anthropic_key_goes_direct(self):
        merged = _user_key_env("anthropic", "sk-ant-user")
        assert merged["ANTHROPIC_API_KEY"] == "sk-ant-user"
        assert merged["ANTHROPIC_AUTH_TOKEN"] == ""
        assert merged["ANTHROPIC_BASE_URL"] == ""
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert merged["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == ""

    def test_openrouter_key_goes_to_openrouter(self):
        merged = _user_key_env("open_router", "sk-or-user")
        assert merged["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert merged["ANTHROPIC_AUTH_TOKEN"] == "sk-or-user"
        assert merged["ANTHROPIC_API_KEY"] == ""
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert merged["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == ""

    @pytest.mark.parametrize("provider", ["anthropic", "open_router"])
    def test_empty_key_is_refused(self, provider):
        with pytest.raises(RuntimeError):
            _user_key_env(provider, "")

    @pytest.mark.parametrize(
        "provider, expected", [("open_router", "0"), ("anthropic", None)]
    )
    def test_billing_header_off_only_off_first_party(self, provider, expected):
        merged = _user_key_env(provider, "user-key")
        assert merged.get("CLAUDE_CODE_ATTRIBUTION_HEADER") == expected
