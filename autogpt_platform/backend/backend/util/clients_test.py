"""Tests for get_openai_client prefer_openrouter parameter."""

from unittest.mock import MagicMock, patch

import pytest

from backend.util.clients import _get_local_openai_client, get_openai_client


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
