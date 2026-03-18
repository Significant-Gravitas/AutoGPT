"""Tests for integration_creds — TTL cache and token lookup paths."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.integration_creds import (
    _NULL_CACHE_TTL,
    _TOKEN_CACHE_TTL,
    PROVIDER_ENV_VARS,
    _null_cache,
    _token_cache,
    get_integration_env_vars,
    get_provider_token,
    invalidate_user_provider_cache,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials

_USER = "user-integration-creds-test"
_PROVIDER = "github"


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
def clear_caches():
    """Ensure clean caches before and after every test."""
    _token_cache.clear()
    _null_cache.clear()
    yield
    _token_cache.clear()
    _null_cache.clear()


class TestInvalidateUserProviderCache:
    def test_removes_token_entry(self):
        key = (_USER, _PROVIDER)
        _token_cache[key] = "tok"
        invalidate_user_provider_cache(_USER, _PROVIDER)
        assert key not in _token_cache

    def test_removes_null_entry(self):
        key = (_USER, _PROVIDER)
        _null_cache[key] = True
        invalidate_user_provider_cache(_USER, _PROVIDER)
        assert key not in _null_cache

    def test_noop_when_key_not_cached(self):
        # Should not raise even when there is no cache entry.
        invalidate_user_provider_cache("no-such-user", _PROVIDER)

    def test_only_removes_targeted_key(self):
        other_key = ("other-user", _PROVIDER)
        _token_cache[other_key] = "other-tok"
        invalidate_user_provider_cache(_USER, _PROVIDER)
        assert other_key in _token_cache


class TestGetProviderToken:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_cached_token_without_db_hit(self):
        _token_cache[(_USER, _PROVIDER)] = "cached-tok"

        mock_manager = MagicMock()
        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "cached-tok"
        mock_manager.store.get_creds_by_provider.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_returns_none_for_null_cached_provider(self):
        _null_cache[(_USER, _PROVIDER)] = True

        mock_manager = MagicMock()
        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        mock_manager.store.get_creds_by_provider.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_api_key_creds_returned_and_cached(self):
        api_creds = _make_api_key_creds("my-api-key")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[api_creds])

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "my-api-key"
        assert _token_cache.get((_USER, _PROVIDER)) == "my-api-key"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth2_preferred_over_api_key(self):
        oauth_creds = _make_oauth2_creds("oauth-tok")
        api_creds = _make_api_key_creds("api-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(
            return_value=[api_creds, oauth_creds]
        )
        mock_manager.refresh_if_needed = AsyncMock(return_value=oauth_creds)

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result == "oauth-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_oauth2_refresh_failure_returns_none(self):
        """On refresh failure, return None instead of caching a stale token."""
        oauth_creds = _make_oauth2_creds("stale-oauth-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[oauth_creds])
        mock_manager.refresh_if_needed = AsyncMock(side_effect=RuntimeError("network"))

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        # Stale tokens must NOT be returned — forces re-auth.
        assert result is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_credentials_caches_null_entry(self):
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[])

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        assert _null_cache.get((_USER, _PROVIDER)) is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_db_exception_returns_none_without_caching(self):
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(
            side_effect=RuntimeError("db down")
        )

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        assert result is None
        # DB errors are not cached — next call will retry
        assert (_USER, _PROVIDER) not in _token_cache
        assert (_USER, _PROVIDER) not in _null_cache

    @pytest.mark.asyncio(loop_scope="session")
    async def test_null_cache_has_shorter_ttl_than_token_cache(self):
        """Verify the TTL constants are set correctly for each cache."""
        assert _null_cache.ttl == _NULL_CACHE_TTL
        assert _token_cache.ttl == _TOKEN_CACHE_TTL
        assert _NULL_CACHE_TTL < _TOKEN_CACHE_TTL


class TestGetIntegrationEnvVars:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_injects_all_env_vars_for_provider(self):
        _token_cache[(_USER, "github")] = "gh-tok"

        result = await get_integration_env_vars(_USER)

        for var in PROVIDER_ENV_VARS["github"]:
            assert result[var] == "gh-tok"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_empty_dict_when_no_credentials(self):
        _null_cache[(_USER, "github")] = True

        result = await get_integration_env_vars(_USER)

        assert result == {}
