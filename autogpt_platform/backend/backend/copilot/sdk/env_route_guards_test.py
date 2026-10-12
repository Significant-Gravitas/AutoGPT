"""Tests for the route guards in ``build_sdk_env`` and ``build_user_key_sdk_env``:
the billing-header switch and the orchestrator block's launch credentials."""

from unittest.mock import patch

import pytest

from .env_test import _make_config

# ---------------------------------------------------------------------------
# Billing-header switch
# ---------------------------------------------------------------------------

_ROUTES = ("openrouter", "direct", "subscription", "codex")


def _env_for_route(route: str, **switches) -> dict[str, str]:
    """``build_sdk_env()`` for *route* with the given config switches."""
    overrides = {
        "openrouter": dict(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        ),
        "direct": dict(use_openrouter=False),
        "subscription": dict(use_claude_code_subscription=True),
        "codex": dict(use_openrouter=False),
    }[route]
    gateway = (
        dict(codex_gateway_url="http://127.0.0.1:43210", codex_gateway_token="t")
        if route == "codex"
        else {}
    )
    cfg = _make_config(**overrides, **switches)
    with (
        patch("backend.copilot.sdk.env.config", cfg),
        patch("backend.copilot.sdk.env.validate_subscription"),
        patch.dict("backend.copilot.sdk.env.os.environ", {}, clear=True),
    ):
        from backend.copilot.sdk.env import build_sdk_env

        return build_sdk_env(**gateway)


class TestAttributionHeaderSwitch:
    """``claude_agent_omit_attribution_header`` removes the billing-header
    system block only where the CLI does not talk to Anthropic first-party."""

    @pytest.mark.parametrize("route", _ROUTES)
    def test_off_by_default(self, route):
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" not in _env_for_route(route)

    @pytest.mark.parametrize(
        "route, expected",
        [
            ("openrouter", "0"),
            ("codex", "0"),
            ("direct", None),
            ("subscription", None),
        ],
    )
    def test_on_sets_env_on_non_first_party_routes(self, route, expected):
        result = _env_for_route(route, claude_agent_omit_attribution_header=True)
        assert result.get("CLAUDE_CODE_ATTRIBUTION_HEADER") == expected


class TestCredentialAndWindowComeFromOneRoute:
    """A launch's credential (``route_credentials``) and its window pin
    (``context_window``) follow the same route decision: no route takes one
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
# build_user_key_sdk_env — the orchestrator block's launch
# ---------------------------------------------------------------------------

_PARENT_WITH_OAUTH = {
    "ANTHROPIC_BASE_URL": "https://remote-gateway.example.com",
    "ANTHROPIC_API_KEY": "sk-ant-deployment-key",
    "ANTHROPIC_AUTH_TOKEN": "inherited-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "inherited-refresh",
}


def _user_key_env(provider: str, user_key: str, **overrides) -> dict[str, str]:
    """The env the CLI sees: the parent's, under the launch's own."""
    cfg = _make_config(**overrides)
    with (
        patch("backend.copilot.sdk.env.config", cfg),
        patch.dict("backend.copilot.sdk.env.os.environ", _PARENT_WITH_OAUTH, True),
    ):
        from backend.copilot.sdk.env import build_user_key_sdk_env

        return {**_PARENT_WITH_OAUTH, **build_user_key_sdk_env(provider, user_key)}


class TestBuildUserKeySdkEnv:
    """A launch on a user's own key carries that key and nothing else."""

    def test_anthropic_key_goes_direct(self):
        merged = _user_key_env("anthropic", "sk-ant-user")
        assert merged["ANTHROPIC_API_KEY"] == "sk-ant-user"
        assert merged["ANTHROPIC_AUTH_TOKEN"] == ""
        assert merged["ANTHROPIC_BASE_URL"] == ""
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert merged["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == ""

    def test_openrouter_key_uses_the_deployment_base_url(self):
        merged = _user_key_env(
            "open_router",
            "sk-or-user",
            use_openrouter=True,
            api_key="sk-or-deployment",
            base_url="https://openrouter.ai/api/v1",
        )
        assert merged["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert merged["ANTHROPIC_AUTH_TOKEN"] == "sk-or-user"
        assert merged["ANTHROPIC_API_KEY"] == "sk-or-user"
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert merged["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == ""

    def test_openrouter_falls_back_to_openrouter_without_a_base_url(self):
        merged = _user_key_env("open_router", "sk-or-user", base_url=None)
        assert merged["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"

    def test_empty_key_is_refused(self):
        with pytest.raises(RuntimeError):
            _user_key_env("anthropic", "")

    @pytest.mark.parametrize(
        "provider, expected", [("open_router", "0"), ("anthropic", None)]
    )
    def test_attribution_switch_only_off_first_party(self, provider, expected):
        merged = _user_key_env(
            provider, "user-key", claude_agent_omit_attribution_header=True
        )
        assert merged.get("CLAUDE_CODE_ATTRIBUTION_HEADER") == expected
