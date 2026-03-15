"""Tests for integration_creds — TTL cache, sentinel, and token lookup paths."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.integration_creds import (
    _CACHE_MAX_SIZE,
    _NO_TOKEN,
    _NULL_CACHE_TTL,
    _TOKEN_CACHE_TTL,
    PROVIDER_ENV_VARS,
    _cache_set,
    _token_cache,
    get_integration_env_vars,
    get_provider_token,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials

_USER = "user-integration-creds-test"
_PROVIDER = "github"


def _clear_cache() -> None:
    _token_cache.clear()


def _make_api_key_creds(key: str = "test-api-key") -> APIKeyCredentials:
    return APIKeyCredentials(
        id="creds-api-key",
        provider=_PROVIDER,
        api_key=SecretStr(key),
        title="Test API Key",
        expires_at=None,
    )


def _make_oauth2_creds(token: str = "test-oauth-token") -> OAuth2Credentials:
    return OAuth2Credentials(
        id="creds-oauth2",
        provider=_PROVIDER,
        title="Test OAuth",
        access_token=SecretStr(token),
        refresh_token=SecretStr("test-refresh"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=[],
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure a clean cache before and after every test."""
    _clear_cache()
    yield
    _clear_cache()


class TestCacheSet:
    def test_inserts_entry(self):
        key = (_USER, _PROVIDER)
        _cache_set(key, "tok", _TOKEN_CACHE_TTL)
        assert key in _token_cache
        value, _ = _token_cache[key]
        assert value == "tok"

    def test_updates_existing_key_without_eviction(self):
        key = (_USER, _PROVIDER)
        _cache_set(key, "tok1", _TOKEN_CACHE_TTL)
        _cache_set(key, "tok2", _TOKEN_CACHE_TTL)
        assert len(_token_cache) == 1
        value, _ = _token_cache[key]
        assert value == "tok2"

    def test_evicts_oldest_when_full(self):
        # Fill to max
        for i in range(_CACHE_MAX_SIZE):
            _cache_set((f"user-{i}", _PROVIDER), f"tok-{i}", _TOKEN_CACHE_TTL)
        assert len(_token_cache) == _CACHE_MAX_SIZE

        # Adding one more should evict the oldest ("user-0")
        _cache_set(("user-new", _PROVIDER), "tok-new", _TOKEN_CACHE_TTL)
        assert len(_token_cache) == _CACHE_MAX_SIZE
        assert ("user-0", _PROVIDER) not in _token_cache
        assert ("user-new", _PROVIDER) in _token_cache


class TestGetProviderToken:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_cached_token_without_db_hit(self):
        key = (_USER, _PROVIDER)
        _cache_set(key, "cached-tok", _TOKEN_CACHE_TTL)

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager"
        ) as MockManager:
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "cached-tok"
        MockManager.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_none_for_cached_sentinel(self):
        key = (_USER, _PROVIDER)
        _cache_set(key, _NO_TOKEN, _NULL_CACHE_TTL)

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager"
        ) as MockManager:
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        MockManager.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_expired_cache_entry_refetches(self):
        key = (_USER, _PROVIDER)
        # Plant an already-expired entry
        _token_cache[key] = ("old-tok", time.monotonic() - 1)

        api_creds = _make_api_key_creds("fresh-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[api_creds])

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "fresh-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_api_key_creds_returned_and_cached(self):
        api_creds = _make_api_key_creds("my-api-key")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[api_creds])

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "my-api-key"
        value, _ = _token_cache[(_USER, _PROVIDER)]
        assert value == "my-api-key"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth2_preferred_over_api_key(self):
        oauth_creds = _make_oauth2_creds("oauth-tok")
        api_creds = _make_api_key_creds("api-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(
            return_value=[api_creds, oauth_creds]
        )
        mock_manager.refresh_if_needed = AsyncMock(return_value=oauth_creds)

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "oauth-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth2_refresh_failure_falls_back_to_stale_token(self):
        oauth_creds = _make_oauth2_creds("stale-oauth-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[oauth_creds])
        mock_manager.refresh_if_needed = AsyncMock(side_effect=RuntimeError("network"))

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "stale-oauth-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_credentials_caches_sentinel_with_short_ttl(self):
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[])

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        value, expires_at = _token_cache[(_USER, _PROVIDER)]
        assert value is _NO_TOKEN
        # Sentinel should expire sooner than a real token TTL
        remaining = expires_at - time.monotonic()
        assert remaining <= _NULL_CACHE_TTL
        assert remaining > _NULL_CACHE_TTL - 5  # within 5s of expected

    @pytest.mark.asyncio(loop_scope="session")
    async def test_db_exception_returns_none_without_caching(self):
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(
            side_effect=RuntimeError("db down")
        )

        with patch(
            "backend.copilot.integration_creds.IntegrationCredentialsManager",
            return_value=mock_manager,
        ):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        # DB errors are not cached — next call will retry
        assert (_USER, _PROVIDER) not in _token_cache


class TestGetIntegrationEnvVars:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_injects_all_env_vars_for_provider(self):
        _cache_set((_USER, "github"), "gh-tok", _TOKEN_CACHE_TTL)

        result = await get_integration_env_vars(_USER)

        for var in PROVIDER_ENV_VARS["github"]:
            assert result[var] == "gh-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_dict_when_no_credentials(self):
        _cache_set((_USER, "github"), _NO_TOKEN, _NULL_CACHE_TTL)

        result = await get_integration_env_vars(_USER)

        assert result == {}
