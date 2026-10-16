"""Unit tests for ``transport_routing.routing_kwargs_for_chat_transport``.

Each test wires up a fresh ``ChatConfig`` and ``Settings`` and asserts
that the resolved ``ProviderRoutingKwargs`` matches the platform route's row
plus the right runtime credential. These pin the integration boundary
between the route table and our centralized
``util/llm/providers.py::call_provider``, so regressions in either side
surface immediately.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.transport_routing import (
    ProviderRoutingKwargs,
    routing_kwargs_for_chat_transport,
)

# Leftover ``CHAT_*`` / ``*_API_KEY`` env vars in the developer or CI
# environment would change the route and break these tests deterministically.
_ENV_VARS_TO_CLEAR = (
    "CHAT_PLATFORM_ROUTE",
    "CHAT_OPENROUTER_API_KEY",
    "CHAT_OPENROUTER_BASE_URL",
    "CHAT_ANTHROPIC_API_KEY",
    "CHAT_ANTHROPIC_BASE_URL",
    "CHAT_LOCAL_BASE_URL",
    "CHAT_LOCAL_API_KEY",
    "CHAT_LOCAL_ADAPTER",
    "CHAT_LOCAL_CONTEXT_TOKENS",
    "CHAT_STANDARD_MODEL",
    "CHAT_ADVANCED_MODEL",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def _patch_chat_cfg(monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig) -> None:
    """Swap the module-level ``copilot.sdk.env.config`` for a test config.

    The helper imports ``chat_cfg`` lazily inside
    ``routing_kwargs_for_chat_transport``, so we patch the singleton
    directly instead of monkeypatching the function-local name.
    """
    from backend.copilot.sdk import env

    monkeypatch.setattr(env, "config", cfg)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **secrets) -> None:
    """Patch ``backend.util.settings.Settings`` to expose the given secrets.

    Only the fields the helper reads (``anthropic_api_key``,
    ``open_router_api_key``) need to be set per-test.
    """
    from backend.util import settings as settings_mod

    fake_secrets = MagicMock()
    fake_secrets.anthropic_api_key = secrets.get("anthropic_api_key", "")
    fake_secrets.open_router_api_key = secrets.get("open_router_api_key", "")
    fake_settings = MagicMock()
    fake_settings.secrets = fake_secrets
    monkeypatch.setattr(settings_mod, "Settings", lambda: fake_settings)
    # ``ChatConfig.batch_anthropic_api_key`` reads the module-level settings.
    from backend.copilot import config as config_mod

    monkeypatch.setattr(config_mod, "settings", fake_settings)


def _local(**overrides) -> ChatConfig:
    return ChatConfig(
        **{
            "platform_route": "local",
            "local_base_url": "http://localhost:11434/v1",
            "local_adapter": "openai_chat",
            "local_context_tokens": 131_072,
            "standard_model": "llama3.1:8b",
            "advanced_model": "llama3.1:8b",
            **overrides,
        }
    )


class TestLocalRoute:
    """The local route dispatches to the operator's server with its own key
    and URL; nothing falls back to an integration key."""

    def test_local_routing_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        _patch_chat_cfg(monkeypatch, _local(local_api_key="ollama-placeholder"))
        _patch_settings(monkeypatch, open_router_api_key="or-secrets-key")

        result = routing_kwargs_for_chat_transport()

        assert result == ProviderRoutingKwargs(
            provider="ollama",
            api_key="ollama-placeholder",
            base_url="http://localhost:11434/v1",
            supports_flex=False,
            cost_log_provider="local",
        )

    def test_a_server_without_a_key_gets_a_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Ollama doesn't validate the bearer token, and CHAT_LOCAL_API_KEY is
        optional: an unset key is sent as a placeholder, never as an
        integration key."""
        _patch_chat_cfg(monkeypatch, _local())
        _patch_settings(monkeypatch, open_router_api_key="or-secrets-key")

        result = routing_kwargs_for_chat_transport()
        assert result.provider == "ollama"
        assert result.api_key == "local"
        assert result.base_url == "http://localhost:11434/v1"
        assert result.supports_flex is False

    def test_anthropic_passthrough_gets_the_openai_compatible_v1(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """With the passthrough adapter CHAT_LOCAL_BASE_URL is the server's
        root; side calls use its OpenAI-compatible /v1."""
        _patch_chat_cfg(
            monkeypatch,
            _local(
                local_base_url="http://localhost:11434",
                local_adapter="anthropic_passthrough",
            ),
        )
        _patch_settings(monkeypatch)

        assert routing_kwargs_for_chat_transport().base_url == (
            "http://localhost:11434/v1"
        )


class TestOpenRouterRoute:
    """OpenRouter: the route's own key, sent to the route's OpenAI-compatible
    URL; never the OpenRouter integration key."""

    def test_the_routes_key_and_url(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(
            platform_route="openrouter",
            openrouter_api_key="or-key-from-chat-config",
            openrouter_base_url="https://or.proxy.example/api/",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, open_router_api_key="or-key-from-secrets")

        assert routing_kwargs_for_chat_transport() == ProviderRoutingKwargs(
            provider="open_router",
            api_key="or-key-from-chat-config",
            base_url="https://or.proxy.example/api/v1",
            supports_flex=True,
            cost_log_provider="open_router",
        )

    def test_the_default_url_is_openrouters_own(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        assert routing_kwargs_for_chat_transport().base_url == (
            "https://openrouter.ai/api/v1"
        )

    def test_never_the_integration_key(self, monkeypatch: pytest.MonkeyPatch):
        _patch_chat_cfg(monkeypatch, ChatConfig(platform_route="openrouter"))
        _patch_settings(monkeypatch, open_router_api_key="or-key-from-secrets")

        assert routing_kwargs_for_chat_transport().api_key == ""


class TestSubscriptionRoute:
    """A Claude login cannot call the Messages API, and no integration key
    stands in for it: background calls have no route at all."""

    def test_no_platform_api_whatever_keys_are_around(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = ChatConfig(
            platform_route="subscription", anthropic_api_key="ant-key-direct-cfg"
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(
            monkeypatch,
            anthropic_api_key="ant-key-from-secrets",
            open_router_api_key="or-key-from-secrets",
        )

        result = routing_kwargs_for_chat_transport()
        assert result.provider is None
        assert result.api_key == ""
        assert result.base_url is None
        assert cfg.batch_anthropic_api_key is None


class TestDirectAnthropicRoute:
    """direct_anthropic: the route's own key, sent to the route's Messages
    API root. The Batch API keeps its own policy (the route's key, else the
    integration key); the synchronous path never takes the integration key."""

    def test_the_routes_key_and_url(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(
            platform_route="direct_anthropic",
            anthropic_api_key="ant-key",
            anthropic_base_url="https://anthropic.proxy.example/",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        assert routing_kwargs_for_chat_transport() == ProviderRoutingKwargs(
            provider="anthropic",
            api_key="ant-key",
            base_url="https://anthropic.proxy.example",
            supports_flex=False,
            cost_log_provider="anthropic",
        )

    def test_the_default_url_is_anthropics_own(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(platform_route="direct_anthropic", anthropic_api_key="k")
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        assert (
            routing_kwargs_for_chat_transport().base_url == "https://api.anthropic.com"
        )

    def test_never_the_integration_key(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(platform_route="direct_anthropic")
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, anthropic_api_key="ant-key-from-secrets")

        assert routing_kwargs_for_chat_transport().api_key == ""
        assert cfg.batch_anthropic_api_key == "ant-key-from-secrets"


class TestDisabledRoute:
    def test_no_route_gives_callers_nothing_to_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_chat_cfg(monkeypatch, ChatConfig(platform_route="disabled"))
        _patch_settings(monkeypatch, open_router_api_key="or-secrets-key")

        result = routing_kwargs_for_chat_transport()
        assert result.provider is None
        assert result.api_key == ""
        assert result.cost_log_provider == "none"
