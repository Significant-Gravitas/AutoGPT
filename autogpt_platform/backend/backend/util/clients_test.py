"""Tests for get_openai_client prefer_openrouter parameter."""

from unittest.mock import MagicMock, patch

import pytest

from backend.util.clients import (
    _get_local_openai_client,
    get_openai_client,
    openrouter_helper_cost_provider,
)


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Clear the @cached client singletons between tests.

    ``_get_local_openai_client`` is a separate ``@cached`` instance from
    ``get_openai_client``; clearing only the latter leaves a stale local
    ``AsyncOpenAI`` aimed at a previous endpoint after a test monkeypatches
    transport/URL. Clear both.
    """
    get_openai_client.cache_clear()
    _get_local_openai_client.cache_clear()
    yield
    get_openai_client.cache_clear()
    _get_local_openai_client.cache_clear()


def _mock_secrets(*, openai_key: str = "", openrouter_key: str = "") -> MagicMock:
    secrets = MagicMock()
    secrets.openai_internal_api_key = openai_key
    secrets.open_router_api_key = openrouter_key
    return secrets


class TestGetOpenaiClientDefault:
    def test_prefers_openai_key(self):
        secrets = _mock_secrets(openai_key="sk-openai", openrouter_key="sk-or")
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            client = get_openai_client()
        assert client is not None
        assert client.api_key == "sk-openai"
        assert "openrouter" not in str(client.base_url or "")

    def test_falls_back_to_openrouter(self):
        secrets = _mock_secrets(openrouter_key="sk-or")
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            client = get_openai_client()
        assert client is not None
        assert client.api_key == "sk-or"

    def test_returns_none_when_no_keys(self):
        secrets = _mock_secrets()
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            assert get_openai_client() is None


class TestGetOpenaiClientPreferOpenrouter:
    def test_returns_openrouter_client(self):
        secrets = _mock_secrets(openai_key="sk-openai", openrouter_key="sk-or")
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            client = get_openai_client(prefer_openrouter=True)
        assert client is not None
        assert client.api_key == "sk-or"

    def test_returns_none_without_openrouter_key(self):
        secrets = _mock_secrets(openai_key="sk-openai")
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            assert get_openai_client(prefer_openrouter=True) is None

    def test_returns_none_when_no_keys(self):
        secrets = _mock_secrets()
        with patch("backend.util.clients.settings") as mock_settings:
            mock_settings.secrets = secrets
            assert get_openai_client(prefer_openrouter=True) is None


class TestOpenrouterHelperCostProvider:
    """``openrouter_helper_cost_provider`` labels the cost row by the endpoint
    ``get_openai_client(prefer_openrouter=True)`` actually routes to, not the
    chat transport identity."""

    def _patch_transport(self, monkeypatch, cfg):
        monkeypatch.setattr("backend.copilot.sdk.env.config", cfg)

    def test_local_transport_is_ollama(self, monkeypatch):
        from backend.copilot.config import ChatConfig

        cfg = ChatConfig(use_local=True, api_key="ollama", base_url="http://h:11434/v1")
        self._patch_transport(monkeypatch, cfg)
        assert openrouter_helper_cost_provider() == "ollama"

    def test_subscription_transport_is_open_router(self, monkeypatch):
        """Regression: subscription routes the helper through OpenRouter when
        OPEN_ROUTER_API_KEY is present — it must not be logged as ``anthropic``
        (the transport's ``cost_log_provider``)."""
        from backend.copilot.config import ChatConfig

        cfg = ChatConfig(
            use_local=False,
            use_claude_code_subscription=True,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        self._patch_transport(monkeypatch, cfg)
        assert openrouter_helper_cost_provider() == "open_router"

    def test_direct_anthropic_transport_is_open_router(self, monkeypatch):
        from backend.copilot.config import ChatConfig

        cfg = ChatConfig(
            use_local=False,
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key=None,
            base_url=None,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        self._patch_transport(monkeypatch, cfg)
        assert openrouter_helper_cost_provider() == "open_router"

    def test_openrouter_transport_is_open_router(self, monkeypatch):
        from backend.copilot.config import ChatConfig

        cfg = ChatConfig(
            use_local=False,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        self._patch_transport(monkeypatch, cfg)
        assert openrouter_helper_cost_provider() == "open_router"
