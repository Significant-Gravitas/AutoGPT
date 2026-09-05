"""Tests for integration_creds — TTL cache and token lookup paths."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.integration_creds import (
    _NULL_CACHE_TTL,
    _TOKEN_CACHE_TTL,
    PROVIDER_ENV_VARS,
    _consume_creds_changed_events,
    _gh_identity_cache,
    _gh_identity_null_cache,
    _null_cache,
    _token_cache,
    get_integration_env_vars,
    get_provider_token,
    invalidate_user_provider_cache,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_events import CredentialsChangedEvent
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    register_creds_changed_hook,
    unregister_creds_changed_hook,
)

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
def no_listener_thread():
    """Keep the lazy subscription from spawning a Redis-connecting thread."""
    with patch("backend.copilot.integration_creds._ensure_cache_invalidation_listener"):
        yield


@pytest.fixture(autouse=True)
def clear_caches():
    """Ensure clean caches before and after every test."""
    _token_cache.clear()
    _null_cache.clear()
    _gh_identity_cache.clear()
    _gh_identity_null_cache.clear()
    yield
    _token_cache.clear()
    _null_cache.clear()
    _gh_identity_cache.clear()
    _gh_identity_null_cache.clear()


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

    def test_clears_gh_identity_cache_for_github_provider(self):
        """When provider is 'github', identity caches must also be cleared."""
        _gh_identity_cache[_USER] = {
            "GIT_AUTHOR_NAME": "Old Name",
            "GIT_AUTHOR_EMAIL": "old@example.com",
            "GIT_COMMITTER_NAME": "Old Name",
            "GIT_COMMITTER_EMAIL": "old@example.com",
        }
        invalidate_user_provider_cache(_USER, "github")
        assert _USER not in _gh_identity_cache

    def test_clears_gh_identity_null_cache_for_github_provider(self):
        """When provider is 'github', the identity null-cache must also be cleared."""
        _gh_identity_null_cache[_USER] = True
        invalidate_user_provider_cache(_USER, "github")
        assert _USER not in _gh_identity_null_cache

    def test_does_not_clear_gh_identity_cache_for_other_providers(self):
        """When provider is NOT 'github', identity caches must be left alone."""
        _gh_identity_cache[_USER] = {
            "GIT_AUTHOR_NAME": "Some Name",
            "GIT_AUTHOR_EMAIL": "some@example.com",
            "GIT_COMMITTER_NAME": "Some Name",
            "GIT_COMMITTER_EMAIL": "some@example.com",
        }
        invalidate_user_provider_cache(_USER, "some-other-provider")
        assert _USER in _gh_identity_cache


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
    async def test_oauth2_refresh_failure_returns_none_without_null_cache(self):
        """On refresh failure, return None but do NOT cache in null_cache.

        The user has credentials — they just couldn't be refreshed right now
        (e.g. transient network error or event-loop mismatch in the copilot
        executor).  Caching a negative result would block all credential
        lookups for 60 s even though the creds exist and may refresh fine
        on the next attempt.
        """
        oauth_creds = _make_oauth2_creds("stale-oauth-tok")
        mock_manager = MagicMock()
        mock_manager.store.get_creds_by_provider = AsyncMock(return_value=[oauth_creds])
        mock_manager.refresh_if_needed = AsyncMock(side_effect=RuntimeError("network"))

        with patch("backend.copilot.integration_creds._manager", mock_manager):
            result = await get_provider_token(_USER, _PROVIDER)

        # Stale tokens must NOT be returned — forces re-auth.
        assert result is None
        # Must NOT cache negative result when refresh failed — next call retries.
        assert (_USER, _PROVIDER) not in _null_cache

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
    async def test_token_cache_ttl_is_the_stale_token_ceiling(self):
        """The TTL is the fallback when a pub/sub invalidation is lost."""
        assert _null_cache.ttl == _NULL_CACHE_TTL
        assert _token_cache.ttl == _TOKEN_CACHE_TTL
        assert _TOKEN_CACHE_TTL <= 60.0


class TestThreadSafetyLocks:
    """Bug reproduction: shared AsyncRedisKeyedMutex across threads caused
    'Future attached to a different loop' when copilot workers accessed
    credentials from different event loops."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_store_locks_returns_per_thread_instance(self):
        """IntegrationCredentialsStore.locks() must return different instances
        for different threads (via @thread_cached)."""
        import asyncio
        import concurrent.futures

        from backend.integrations.credentials_store import IntegrationCredentialsStore

        store = IntegrationCredentialsStore()

        async def get_locks_id():
            mock_redis = AsyncMock()
            with patch(
                "backend.integrations.credentials_store.get_redis_async",
                return_value=mock_redis,
            ):
                locks = await store.locks()
                return id(locks)

        # Get locks from main thread
        main_id = await get_locks_id()

        # Get locks from a worker thread
        def run_in_thread():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(get_locks_id())
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            worker_id = await asyncio.get_event_loop().run_in_executor(
                pool, run_in_thread
            )

        assert main_id != worker_id, (
            "Store.locks() returned the same instance across threads. "
            "This would cause 'Future attached to a different loop' errors."
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_manager_delegates_to_store_locks(self):
        """IntegrationCredentialsManager.locks() should delegate to store."""
        from backend.integrations.creds_manager import IntegrationCredentialsManager

        manager = IntegrationCredentialsManager()
        mock_redis = AsyncMock()

        with patch(
            "backend.integrations.credentials_store.get_redis_async",
            return_value=mock_redis,
        ):
            locks = await manager.locks()

        # Should have gotten it from the store
        assert locks is not None


class TestRefreshUnlockedPath:
    """Bug reproduction: copilot worker threads need lock-free refresh because
    Redis-backed asyncio.Lock created on one event loop can't be used on another."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_refresh_if_needed_lock_false_skips_redis(self):
        """refresh_if_needed(lock=False) must not touch Redis locks at all."""
        from backend.integrations.creds_manager import IntegrationCredentialsManager

        manager = IntegrationCredentialsManager()
        creds = _make_oauth2_creds()

        mock_handler = MagicMock()
        mock_handler.needs_refresh = MagicMock(return_value=False)
        mock_handler.ROTATES_REFRESH_TOKEN = False

        with patch(
            "backend.integrations.creds_manager._get_provider_oauth_handler",
            new_callable=AsyncMock,
            return_value=mock_handler,
        ):
            result = await manager.refresh_if_needed(_USER, creds, lock=False)

        # Should return credentials without touching locks
        assert result.id == creds.id


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


class TestCacheInvalidationListener:
    """The cross-process half: a write elsewhere reaches this process's cache."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_drops_exactly_the_affected_entry(self):
        _token_cache[(_USER, "github")] = "stale"
        _token_cache[(_USER, "notion")] = "untouched"
        _token_cache[("other-user", "github")] = "untouched"

        await _consume(CredentialsChangedEvent(user_id=_USER, provider="github"))

        assert (_USER, "github") not in _token_cache
        assert _token_cache[(_USER, "notion")] == "untouched"
        assert _token_cache[("other-user", "github")] == "untouched"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_resubscribes_when_the_stream_ends(self):
        """A closed subscription must not leave the cache unwatched."""
        streams = 0

        async def stream():
            nonlocal streams
            streams += 1
            if streams > 1:
                raise asyncio.CancelledError
            return
            yield  # pragma: no cover - makes this an async generator

        with patch("backend.copilot.integration_creds.listen_creds_changed", stream):
            with pytest.raises(asyncio.CancelledError):
                await _consume_creds_changed_events()

        assert streams == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_upgrade_after_fetch_is_visible_without_waiting_for_the_ttl(self):
        """OPEN-3472: a token fetched at T, upgraded at T+60s, was still served
        from cache until the 300 s TTL lapsed at T+300s."""
        manager = MagicMock()
        manager.store.get_creds_by_provider = AsyncMock(
            return_value=[_make_api_key_creds("old-scopes-token")]
        )
        with patch("backend.copilot.integration_creds._manager", manager):
            assert await get_provider_token(_USER, _PROVIDER) == "old-scopes-token"

        published = await _write_credential_as_another_process()
        assert _token_cache[(_USER, _PROVIDER)] == "old-scopes-token"

        for event in published:
            await _consume(event)

        manager.store.get_creds_by_provider = AsyncMock(
            return_value=[_make_api_key_creds("new-scopes-token")]
        )
        with patch("backend.copilot.integration_creds._manager", manager):
            assert await get_provider_token(_USER, _PROVIDER) == "new-scopes-token"


async def _consume(*events: CredentialsChangedEvent) -> None:
    """Run the listener over *events*, then stop it."""

    async def stream():
        for event in events:
            yield event
        raise asyncio.CancelledError

    with patch("backend.copilot.integration_creds.listen_creds_changed", stream):
        with pytest.raises(asyncio.CancelledError):
            await _consume_creds_changed_events()


async def _write_credential_as_another_process() -> list[CredentialsChangedEvent]:
    """Perform a credential write with no in-process hook — the API process's
    shape — and return what it broadcast."""
    published: list[CredentialsChangedEvent] = []

    async def capture(user_id: str, provider: str) -> None:
        published.append(CredentialsChangedEvent(user_id=user_id, provider=provider))

    manager = IntegrationCredentialsManager()
    manager.store = MagicMock()
    manager.store.update_creds = AsyncMock()
    manager._locked = lambda *args, **kwargs: contextlib.nullcontext()

    unregister_creds_changed_hook()
    try:
        with patch("backend.integrations.creds_manager.publish_creds_changed", capture):
            await manager.update(_USER, _make_oauth2_creds("new-scopes-token"))
    finally:
        register_creds_changed_hook(invalidate_user_provider_cache)

    return published
